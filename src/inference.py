import os
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_torch
import yaml

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from .data import load_sampled_data
from .model import SelfGatedHierarchicalTransformerEncoder, CosineMarginClassifier
from .diffusion import DecisionSpaceDiffusion

from src.plots import (
    plot_embedding,
    plot_inter_intra_distributions,
    plot_tsne_triplet,
    plot_confusion_matrix_heatmap,
    gates_to_sensor_segment_matrix,
    plot_topk_sensors,
)


def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    p = argparse.ArgumentParser(
        description="Inference: test accuracy + visualizations"
    )
    p.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent.parent / "config.yaml"),
    )
    p.add_argument("--seed", type=int, default=123)

    p.add_argument(
        "--n-gen",
        type=int,
        default=1200,
        help="Number of diffusion embeddings per fault for t-SNE triplets.",
    )

    p.add_argument(
        "--ratio",
        type=int,
        default=5,
        help="Target ratio for tsne plots: "
             "normal = ratio * fault, and generated = (ratio-1) * fault.",
    )

    return p.parse_args()


def sample_indices(mask, max_n, rng):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return idx
    if len(idx) > max_n:
        idx = rng.choice(idx, size=max_n, replace=False)
    return idx


def build_model_from_ckpt(clf_state, input_dim, cfg, device):
   
    d_model = clf_state["input_proj.weight"].shape[0]
    input_dim_ckpt = clf_state["input_proj.weight"].shape[1]
    if input_dim != input_dim_ckpt:
        raise ValueError(
            f"Feature dim mismatch: data={input_dim}, ckpt={input_dim_ckpt}"
        )

    if "classifier.3.weight" in clf_state:
        num_classes = clf_state["classifier.3.weight"].shape[0]
    elif "cos_head.W" in clf_state:
        num_classes = clf_state["cos_head.W"].shape[0]
    else:
        num_classes = None

    low_layers = {
        int(k.split(".")[2])
        for k in clf_state.keys()
        if k.startswith("encoder_low.layers.")
    }
    high_layers = {
        int(k.split(".")[2])
        for k in clf_state.keys()
        if k.startswith("encoder_high.layers.")
    }
    num_layers_low = max(low_layers) + 1 if low_layers else 0
    num_layers_high = max(high_layers) + 1 if high_layers else 0

    dim_feedforward = clf_state["encoder_low.layers.0.linear1.weight"].shape[0]

    feat_dim = (
        clf_state["cos_head.W"].shape[1]
        if "cos_head.W" in clf_state
        else d_model
    )

    model = SelfGatedHierarchicalTransformerEncoder(
        input_dim=input_dim,
        d_model=d_model,
        nhead=4,
        num_layers_low=num_layers_low,
        num_layers_high=num_layers_high,
        dim_feedforward=dim_feedforward,
        dropout=0.05,
        pool_output_size=10,
        num_classes=num_classes,
    ).to(device)

    mcfg = cfg.get("model", {})
    base_m = float(mcfg.get("m", 0.15))
    s_val = float(mcfg.get("s", 16.0))
    margin_type = str(mcfg.get("margin_type", "arc"))

    model.cos_head = CosineMarginClassifier(
        feat_dim=feat_dim,
        num_classes=num_classes,
        s=s_val,
        m=base_m,
        margin_type=margin_type,
    ).to(device)

    per_m = torch.full((num_classes,), base_m, device=device)
    for k, v in (mcfg.get("per_class_margin_overrides", {}) or {}).items():
        idx = int(k)
        if idx < num_classes:
            per_m[idx] = float(v)
    model.cos_head.per_class_margin = per_m

    model.load_state_dict(clf_state, strict=True)
    model.eval()

    return model, feat_dim, num_classes


def extract_feats(model, X, feat_dim, device):
    bs = 256
    all_f = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i : i + bs]).float().to(device)
            f = model.forward_features(xb)
            f = F_torch.normalize(f, dim=-1)
            all_f.append(f.cpu().numpy())
    if not all_f:
        return np.zeros((0, feat_dim), dtype=np.float32)
    return np.concatenate(all_f, axis=0)


def predict_batches(model, X, device):
    preds = []
    bs = 256
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i : i + bs]).float().to(device)
            feats = model.forward_features(xb)
            logits = model.cos_head(feats, y=None, use_margin=False)
            preds.append(logits.argmax(1).cpu().numpy())
    if not preds:
        return np.zeros((0,), dtype=np.int64)
    return np.concatenate(preds)



def main():
    args = parse_args()
    set_seed(args.seed)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    training_cfg = cfg.get("training", {})
    diff_cfg = (training_cfg.get("diffusion", {}) or {})
    use_diffusion = bool(diff_cfg.get("enabled", True))

    ratio = int(args.ratio)

    base_dir = "/workspace/ClassImbalanceAwareTransformer"

    best_ckpt = f"{base_dir}/best_state_dict_{ratio}.pt"
    diff_ckpt = f"{base_dir}/diffusion_state_dict_{ratio}.pt"


    if not os.path.exists(best_ckpt):
        raise FileNotFoundError(f"best_state_dict.pt not found at: {best_ckpt}")
    if use_diffusion and not os.path.exists(diff_ckpt):
        raise FileNotFoundError(f"diffusion_state_dict.pt not found at: {diff_ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # directory for all figures
    fig_dir = "/workspace/ClassImbalanceAwareTransformer/figures"
    os.makedirs(fig_dir, exist_ok=True)


    print("\n=== Loading Testing data for accuracy ===")
    ff_path_test = "/workspace/TEP_FaultFree_Testing.RData"
    ft_path_test = "/workspace/TEP_Faulty_Testing.RData"

    window_size = cfg["data_windowing"]["window_size"]
    stride = cfg["data_windowing"]["stride"]
    post_fault_start = cfg["data_windowing"]["post_fault_start"]

    normal_test_start = cfg["data_windowing"].get("normal_test_start", 10000)
    normal_test_end = cfg["data_windowing"].get("normal_test_end", 20000)

    test_runs_for_accuracy = range(1, 20)
    train_runs_dummy = []

    (X_train_dummy, y_train_dummy, _), (X_test, y_test, _) = load_sampled_data(
        window_size=window_size,
        stride=stride,
        ff_path=ff_path_test,
        ft_path=ft_path_test,
        post_fault_start=post_fault_start,
        train_runs=train_runs_dummy,
        test_runs=test_runs_for_accuracy,
        normal_test_start=normal_test_start,
        normal_test_end=normal_test_end,
    )

    clf_state = torch.load(best_ckpt, map_location=device)
    input_dim = X_test.shape[2]

    model, feat_dim, num_classes = build_model_from_ckpt(
        clf_state, input_dim, cfg, device
    )

    print("\n=== Per fault accuracy on Testing set ===")
    y_pred = predict_batches(model, X_test, device)

    per_fault_acc = {}
    acc_values = []

    n_classes = int(y_test.max()) + 1
    for c in range(n_classes):
        mask = y_test == c
        if mask.sum() == 0:
            acc = float("nan")
        else:
            acc = (y_pred[mask] == c).mean()
            acc_values.append(acc)
        per_fault_acc[c] = acc
        print(f"Fault {c}: {acc:.4f}")

    if acc_values:
        mean_acc = float(np.mean(acc_values))
        print(f"\nAccuracy on Testing set: {mean_acc:.4f}")
    else:
        print("\nMean accuracy could not be computed (no valid classes).")

    
    cm_path = os.path.join(fig_dir, "confusion_matrix_test.png")
    plot_confusion_matrix_heatmap(
        y_test,
        y_pred,
        num_classes=num_classes,
        save_path=cm_path,
    )
    print(f"Saved confusion matrix to {cm_path}")

    ff_path_train = "/workspace/TEP_FaultFree_Training.RData"
    ft_path_train = "/workspace/TEP_Faulty_Training.RData"

    train_runs = range(
        cfg["data_windowing"]["train_runs_start"],
        cfg["data_windowing"]["train_runs_end"],
    )
    test_runs_for_plots = range(
        cfg["data_windowing"]["test_runs_start"],
        cfg["data_windowing"]["test_runs_end"],
    )

    (X_train_vis, y_train_vis, _), (X_val_vis, y_val_vis, _) = load_sampled_data(
        window_size=window_size,
        stride=stride,
        ff_path=ff_path_train,
        ft_path=ft_path_train,
        post_fault_start=post_fault_start,
        train_runs=train_runs,
        test_runs=test_runs_for_plots,
    )
    X_all_vis = np.concatenate([X_train_vis, X_val_vis], axis=0)
    y_all_vis = np.concatenate([y_train_vis, y_val_vis], axis=0)

    feats_all = extract_feats(model, X_all_vis, feat_dim, device)

    print("\n=== Global t-SNE of all classes ===")
    tsne_all_path = os.path.join(fig_dir, "tsne_all_classes.png")
    plot_embedding(
        feats_all,
        y_all_vis,
        method="tsne",
        save_path=tsne_all_path,
        sample_per_class=800,
        seed=args.seed,
    )
    print(f"Saved global t-SNE to {tsne_all_path}")

    print("\n=== Intra vs Inter class distance plot ===")
    C = num_classes
    centers = np.zeros((C, feat_dim), dtype=np.float32)
    for c in range(C):
        m = y_all_vis == c
        if m.any():
            centers[c] = feats_all[m].mean(axis=0)
    inter_intra_path = os.path.join(fig_dir, "inter_intra_distributions.png")
    plot_inter_intra_distributions(
        feats_all, y_all_vis, centers, save_path=inter_intra_path
    )
    print(f"Saved intra vs inter distance plot to {inter_intra_path}")

  

    X_all_vis_tensor = torch.from_numpy(X_all_vis).float().to(device)
    fault_ids_for_interpret = [3, 9, 15]
    sensor_names = [f"Var{i+1}" for i in range(X_all_vis.shape[2])]
    max_windows_per_fault = 512

    for fid in fault_ids_for_interpret:
        idx_fault = np.where(y_all_vis == fid)[0]
        if len(idx_fault) == 0:
            print(f"No windows for fault {fid} in training data, skipping interpretability.")
            continue

        sel = idx_fault[: min(max_windows_per_fault, len(idx_fault))]
        xb = X_all_vis_tensor[sel]

        with torch.no_grad():
            logits, extras = model(xb, return_gates=True)

        # sensor × segment matrix
        M = gates_to_sensor_segment_matrix(extras, reduce="max")

        out_png = os.path.join(fig_dir, f"gating_top10_fault_{fid}.png")
        plot_topk_sensors(
            M,
            sensor_names=sensor_names,
            k=10,
            fault_id=fid,
            out_png=out_png,
            label="Mean Gate Weight (across segments)",
            title_prefix="Top Sensors by Gate Weight",
        )
        print(f"Saved top 10 sensor gating plot for fault {fid} to {out_png}")



    ddm = None
    if use_diffusion:
        print("\n=== Loading diffusion model for generated embeddings ===")
        T = int(diff_cfg.get("T", 1000))
        steps_infer = int(diff_cfg.get("steps_infer", 24))
        width = int(diff_cfg.get("width", 512))
        depth = int(diff_cfg.get("depth", 3))

        ddm = DecisionSpaceDiffusion(
            feat_dim,
            num_classes,
            T=T,
            num_steps_infer=steps_infer,
            width=width,
            depth=depth,
        ).to(device)
        ddm_state = torch.load(diff_ckpt, map_location=device)
        ddm.load_state_dict(ddm_state)
        ddm.eval()
    else:
        print("\nDiffusion disabled in config, plots will not use generated samples.")

   
    faults_to_plot = [3, 9, 15]
    for fid in faults_to_plot:
        print(f"\n=== t-SNE for fault {fid} ===")
        gen_feats = None
        if ddm is not None:
            with torch.no_grad():
                y_gen = torch.full(
                    (args.n_gen,), fid, dtype=torch.long, device=device
                )
                Z_gen = ddm.ddim_sample(
                    y=y_gen,
                    n=args.n_gen,
                    steps=int(diff_cfg.get("steps_infer", 24)),
                )
                gen_feats = Z_gen.cpu().numpy()

        save_path = os.path.join(fig_dir, f"tsne_fault_{fid}.png")
        try:

            plot_tsne_triplet(
                feats_all,
                y_all_vis,
                gen_feats,
                fault_id=fid,
                save_path=save_path,
                seed=args.seed,
                ratio=args.ratio,
            )
            print(f"Saved t-SNE for fault {fid} to {save_path}")
        except ValueError as e:
            print(f"Skipping fault {fid} plot: {e}")



if __name__ == "__main__":
    main()
