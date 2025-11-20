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
from .model import SelfGatedHierarchicalTransformerEncoder
from .diffusion import DecisionSpaceDiffusion  # same class used in training


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    p = argparse.ArgumentParser(description="Inference t-SNE with diffusion generated samples")
    p.add_argument("--config", type=str,
                   default=str(Path(__file__).parent.parent / "config.yaml"))
    p.add_argument("--results-dir", type=str, required=True,
                   help="Folder that contains best_state_dict.pt and diffusion_state_dict.pt")
    p.add_argument("--fault-id", type=int, default=9,
                   help="Fault class to visualize")
    p.add_argument("--n-normal", type=int, default=500,
                   help="Number of normal windows to use")
    p.add_argument("--n-fault", type=int, default=500,
                   help="Number of real fault windows to use")
    p.add_argument("--n-gen", type=int, default=500,
                   help="Number of generated fault embeddings")
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

    results_dir = args.results_dir
    if not os.path.exists(os.path.join(results_dir, "best_state_dict.pt")):
        raise FileNotFoundError("best_state_dict.pt not found in results_dir")
    if not os.path.exists(os.path.join(results_dir, "diffusion_state_dict.pt")):
        raise FileNotFoundError("diffusion_state_dict.pt not found in results_dir")

    # same data loading as training
    ff_path = cfg["dataset"]["ff_path"]
    ft_path = cfg["dataset"]["ft_path"]

    window_size = cfg["data_windowing"]["window_size"]
    stride = cfg["data_windowing"]["stride"]
    post_fault_start = cfg["data_windowing"]["post_fault_start"]

    train_runs = range(cfg["data_windowing"]["train_runs_start"],
                       cfg["data_windowing"]["train_runs_end"])
    test_runs = range(cfg["data_windowing"]["test_runs_start"],
                      cfg["data_windowing"]["test_runs_end"])

    (_, _, _), (X_test, y_test, _) = load_sampled_data(
        window_size=window_size,
        stride=stride,
        ff_path=ff_path,
        ft_path=ft_path,
        post_fault_start=post_fault_start,
        train_runs=train_runs,
        test_runs=test_runs,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build classifier and load weights
    num_classes = int(y_test.max()) + 1
    model = SelfGatedHierarchicalTransformerEncoder(
        input_dim=X_test.shape[2],
        num_classes=num_classes
    ).to(device)
    clf_state = torch.load(os.path.join(results_dir, "best_state_dict.pt"),
                           map_location=device)
    model.load_state_dict(clf_state)
    model.eval()

    # get feature dimension from one forward pass
    with torch.no_grad():
        x0 = torch.from_numpy(X_test[:1]).float().to(device)
        feat_dim = model.forward_features(x0).shape[-1]

    # build diffusion model with same config as training
    diff_cfg = (cfg.get("training", {}).get("diffusion", {}) or {})
    T = int(diff_cfg.get("T", 1000))               # if T not stored, default 1000
    steps_infer = int(diff_cfg.get("steps_infer", 24))
    width = int(diff_cfg.get("width", 512))
    depth = int(diff_cfg.get("depth", 3))

    ddm = DecisionSpaceDiffusion(
        feat_dim,
        num_classes,
        T=T,
        num_steps_infer=steps_infer,
        width=width,
        depth=depth
    ).to(device)
    ddm_state = torch.load(os.path.join(results_dir, "diffusion_state_dict.pt"),
                           map_location=device)
    ddm.load_state_dict(ddm_state)
    ddm.eval()

    rng = np.random.default_rng(args.seed)

    fault_id = args.fault_id
    normal_label = 0

    # select indices for normal and fault from test set
    idx_normal = sample_indices(y_test == normal_label, args.n_normal, rng)
    idx_fault = sample_indices(y_test == fault_id, args.n_fault, rng)

    X_norm = X_test[idx_normal]
    X_fault = X_test[idx_fault]

    # compute classifier features
    def extract_feats(X):
        bs = 256
        all_f = []
        with torch.no_grad():
            for i in range(0, len(X), bs):
                x_batch = torch.from_numpy(X[i:i + bs]).float().to(device)
                f = model.forward_features(x_batch)
                # normalize to match diffusion training space
                f = F.normalize(f, dim=-1)
                all_f.append(f.cpu().numpy())
        if not all_f:
            return np.zeros((0, feat_dim), dtype=np.float32)
        return np.concatenate(all_f, axis=0)

    feats_norm = extract_feats(X_norm)
    feats_fault = extract_feats(X_fault)

    # generate synthetic embeddings for the fault class
    with torch.no_grad():
        y_gen = torch.full((args.n_gen,), fault_id, dtype=torch.long, device=device)
        Z_gen = ddm.ddim_sample(y=y_gen, n=args.n_gen, steps=steps_infer)
        feats_gen = Z_gen.cpu().numpy()

    # prepare data for t-SNE
    X_all = np.vstack([feats_norm, feats_fault, feats_gen])
    labels = (["Normal"] * len(feats_norm) +
              [f"Fault {fault_id}"] * len(feats_fault) +
              [f"Fault {fault_id} Generated"] * len(feats_gen))

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

    plt.title(f"t-SNE: Normal vs Fault {fault_id} vs Generated (inference)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

