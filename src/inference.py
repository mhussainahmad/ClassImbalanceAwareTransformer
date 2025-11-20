import os
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import yaml

from .data import load_sampled_data
from .model import SelfGatedHierarchicalTransformerEncoder, CosineMarginClassifier
from .diffusion import DecisionSpaceDiffusion


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
        description="Inference: per fault accuracy + t-SNE with diffusion samples"
    )
    p.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent.parent / "config.yaml"),
    )
    p.add_argument(
        "--n-normal",
        type=int,
        default=500,
        help="Number of normal windows to use in each t-SNE",
    )
    p.add_argument(
        "--n-fault",
        type=int,
        default=500,
        help="Number of real fault windows to use in each t-SNE",
    )
    p.add_argument(
        "--n-gen",
        type=int,
        default=500,
        help="Number of generated fault embeddings in each t-SNE",
    )
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def sample_indices(mask, max_n, rng):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return idx
    if len(idx) > max_n:
        idx = rng.choice(idx, size=max_n, replace=False)
    return idx


def main():
    args = parse_args()
    set_seed(args.seed)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Hardcoded checkpoints in repo root
    best_ckpt = "/workspace/ClassImbalanceAwareTransformer/best_state_dict.pt"
    diff_ckpt = "/workspace/ClassImbalanceAwareTransformer/diffusion_state_dict.pt"

    if not os.path.exists(best_ckpt):
        raise FileNotFoundError(f"best_state_dict.pt not found at: {best_ckpt}")
    if not os.path.exists(diff_ckpt):
        raise FileNotFoundError(f"diffusion_state_dict.pt not found at: {diff_ckpt}")

    # Figures directory
    fig_dir = "/workspace/ClassImbalanceAwareTransformer/figures"
    os.makedirs(fig_dir, exist_ok=True)

    # === DATA: hardcoded testing files and test slicing ===

    # Hardcoded testing RData paths
    ff_path = "/workspace/TEP_FaultFree_Testing.RData"
    ft_path = "/workspace/TEP_Faulty_Testing.RData"

    # Use windowing settings from config
    window_size = cfg["data_windowing"]["window_size"]
    stride = cfg["data_windowing"]["stride"]
    post_fault_start = cfg["data_windowing"]["post_fault_start"]

    # Hardcoded test selection for fault 0
    normal_test_start = 1
    normal_test_end = 50000
    train_runs_end = 10
    post_fault_start = 160
    test_runs_start: 1
    test_runs_end: 20
    # Hardcoded faulty test runs
    test_runs = range(1, 40)

    # No training runs needed for inference, but load_sampled_data
    # still returns a "train" part that we only use to get input_dim
    train_runs = []

    (X_train, y_train, _), (X_test, y_test, _) = load_sampled_data(
        window_size=window_size,
        stride=stride,
        ff_path=ff_path,
        ft_path=ft_path,
        post_fault_start=post_fault_start,
        train_runs=train_runs,
        test_runs=test_runs,
        normal_test_start=normal_test_start,
        normal_test_end=normal_test_end,
    )

    print("Inference X_test shape:", X_test.shape)
    print("Inference y_test shape:", y_test.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load checkpoint first, to derive num_classes from it ===
    clf_state = torch.load(best_ckpt, map_location=device)

    # classifier.3.weight has shape [num_classes, hidden_dim]
    if "classifier.3.weight" not in clf_state:
        raise KeyError("classifier.3.weight not found in checkpoint; cannot infer num_classes")

    num_classes = clf_state["classifier.3.weight"].shape[0]
    input_dim = X_train.shape[2]

    # === MODEL + COSINE HEAD (same as training) ===
    model = SelfGatedHierarchicalTransformerEncoder(
        input_dim=input_dim,
        num_classes=num_classes,
    ).to(device)

    # get feat_dim from a dummy batch
    with torch.no_grad():
        x0 = torch.from_numpy(X_train[:1]).float().to(device)
        feat_dim = model.forward_features(x0).shape[-1]

    # cosine margin head configuration
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

    # per class margin overrides
    per_m = torch.full((num_classes,), base_m, device=device)
    for k, v in (mcfg.get("per_class_margin_overrides", {}) or {}).items():
        idx = int(k)
        if idx < num_classes:
            per_m[idx] = float(v)
    model.cos_head.per_class_margin = per_m

    # load full checkpoint into model (including cos_head.*)
    model.load_state_dict(clf_state, strict=True)
    model.eval()

    # === DIFFUSION MODEL ===
    diff_cfg = (cfg.get("training", {}).get("diffusion", {}) or {})
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

    rng = np.random.default_rng(args.seed)
    normal_label = 0

    # === helpers ===
    def extract_feats(X):
        """Classifier feature space, normalized, same as diffusion training space."""
        bs = 256
        all_f = []
        with torch.no_grad():
            for i in range(0, len(X), bs):
                xb = torch.from_numpy(X[i:i + bs]).float().to(device)
                f = model.forward_features(xb)
                f = F.normalize(f, dim=-1)
                all_f.append(f.cpu().numpy())
        if not all_f:
            return np.zeros((0, feat_dim), dtype=np.float32)
        return np.concatenate(all_f, axis=0)

    def predict_batches(X):
        """Use cosine margin head for predictions, same as in training evaluate."""
        preds = []
        bs = 256
        with torch.no_grad():
            for i in range(0, len(X), bs):
                xb = torch.from_numpy(X[i:i + bs]).float().to(device)
                feats = model.forward_features(xb)
                logits = model.cos_head(feats, y=None, use_margin=False)
                preds.append(logits.argmax(1).cpu().numpy())
        if not preds:
            return np.zeros((0,), dtype=np.int64)
        return np.concatenate(preds)

 
 
    print("\n=== Per fault accuracy on test set ===")

    y_pred = predict_batches(X_test)

    per_fault_acc = {}
    acc_values = []   # store valid accuracies only

    n_classes = int(y_test.max()) + 1
    for c in range(n_classes):
        mask = (y_test == c)
        if mask.sum() == 0:
            acc = float("nan")
        else:
            acc = (y_pred[mask] == c).mean()
            acc_values.append(acc)

        per_fault_acc[c] = acc
        print(f"Fault {c}: {acc:.4f}")

    # ---- print macro mean accuracy ----
    if len(acc_values) > 0:
        mean_acc = float(np.mean(acc_values))
        print(f"\nAccuracy: {mean_acc:.4f}")
    else:
        print("\nMean accuracy could not be computed (no valid classes).")

     # === t-SNE for faults 3, 9, 15 ===
    faults_to_plot = [3, 9, 15]

    # features for normals shared across plots
    idx_normal = sample_indices(y_test == normal_label, args.n_normal, rng)
    X_norm = X_test[idx_normal]
    feats_norm = extract_feats(X_norm)

    for fault_id in faults_to_plot:
        print(f"\n=== t-SNE for fault {fault_id} ===")
        idx_fault = sample_indices(y_test == fault_id, args.n_fault, rng)
        if len(idx_fault) == 0:
            print(f"No samples for fault {fault_id} in test set, skipping plot.")
            continue

        X_fault = X_test[idx_fault]
        feats_fault = extract_feats(X_fault)

        # generate synthetic embeddings for this fault
        with torch.no_grad():
            y_gen = torch.full((args.n_gen,), fault_id, dtype=torch.long, device=device)
            Z_gen = ddm.ddim_sample(y=y_gen, n=args.n_gen, steps=steps_infer)
            feats_gen = Z_gen.cpu().numpy()

        X_all = np.vstack([feats_norm, feats_fault, feats_gen])

        tsne = TSNE(n_components=2, perplexity=30, random_state=args.seed)
        X_2d = tsne.fit_transform(X_all)

        n_norm = len(feats_norm)
        n_fault = len(feats_fault)
        n_gen = len(feats_gen)

        Xn = X_2d[:n_norm]
        Xf = X_2d[n_norm:n_norm + n_fault]
        Xg = X_2d[n_norm + n_fault:n_norm + n_fault + n_gen]

        plt.figure(figsize=(7, 5))
        plt.scatter(Xn[:, 0], Xn[:, 1], s=10, alpha=0.7, label="Normal")
        plt.scatter(Xf[:, 0], Xf[:, 1], s=10, alpha=0.7, label=f"Fault {fault_id}")
        plt.scatter(Xg[:, 0], Xg[:, 1], s=10, alpha=0.7, label=f"Fault {fault_id} Generated")

        plt.title(f"t-SNE: Normal vs Fault {fault_id} vs Generated")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(fig_dir, f"tsne_fault_{fault_id}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved t-SNE figure for fault {fault_id} to: {out_path}")


if __name__ == "__main__":
    main()
