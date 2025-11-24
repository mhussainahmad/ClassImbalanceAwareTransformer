import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, balanced_accuracy_score, f1_score
)
from sklearn.manifold import TSNE

@torch.no_grad()
def get_preds_and_logits(model, loader, device):
    """Return y_true, y_pred, logits (CPU)."""
    model.eval()
    ys, ps, ls = [], [], []
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        feats  = model.forward_features(x)
        logits = model.cos_head(feats, y=None, use_margin=False)
        preds  = logits.argmax(dim=1)
        ys.append(y.cpu()); ps.append(preds.cpu()); ls.append(logits.detach().cpu())
    y_true = torch.cat(ys).numpy()
    y_pred = torch.cat(ps).numpy()
    logits_all = torch.cat(ls).numpy()
    return y_true, y_pred, logits_all

@torch.no_grad()
def get_features(model, loader, device):
    """Return (feats, y) from forward_features()."""
    model.eval()
    feats_all, ys = [], []
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        feats_all.append(model.forward_features(x).detach().cpu())
        ys.append(y.cpu())
    feats = torch.cat(feats_all).numpy()
    y = torch.cat(ys).numpy()
    return feats, y

def _ensure_dir(p="results"):
    os.makedirs(p, exist_ok=True)
    return p

def plot_class_dist_and_margins(y_train, model,
                                save_path="results/class_dist_margins.png",
                                highlight=(0, 9, 15)):
    _ensure_dir(os.path.dirname(save_path) or ".")
    labels, counts = np.unique(y_train.astype(int), return_counts=True)
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.bar(labels, counts, alpha=0.85)
    ax1.set_xlabel("Class"); ax1.set_ylabel("Train count"); ax1.set_xticks(labels)

    margins = None
    if hasattr(model, "cos_head"):
        if getattr(model.cos_head, "per_class_margin", None) is not None:
            margins = model.cos_head.per_class_margin.detach().cpu().numpy()
        elif hasattr(model.cos_head, "m"):
            margins = np.full(len(labels), float(model.cos_head.m))

    if margins is not None:
        ax2 = ax1.twinx()
        ax2.plot(labels, margins, marker="o", linewidth=2)
        ax2.set_ylabel("Per-class margin $m_c$")
        for c in highlight:
            if c in labels:
                i = int(np.where(labels == c)[0])
                ax1.bar(c, counts[i], color="#d62728", alpha=0.55)
                ax2.plot(c, margins[c], marker="o", color="#d62728")

    plt.title("Class counts and per-class margins")
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close(fig)


def per_class_report(y_true, y_pred):
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    classes = sorted([int(k) for k in rep.keys() if k.isdigit()])
    f1  = np.array([rep[str(c)]["f1-score"]  for c in classes])
    rec = np.array([rep[str(c)]["recall"]    for c in classes])
    pre = np.array([rep[str(c)]["precision"] for c in classes])
    return np.array(classes), pre, rec, f1

def plot_per_class_bars(y_true_base, y_pred_base, y_true_ours, y_pred_ours,
                        metric="recall", save_path="results/per_class_bars.png",
                        highlight=(0, 9, 15)):
    _ensure_dir(os.path.dirname(save_path) or ".")
    classes_b, pre_b, rec_b, f1_b = per_class_report(y_true_base, y_pred_base)
    classes_o, pre_o, rec_o, f1_o = per_class_report(y_true_ours, y_pred_ours)
    assert (classes_b == classes_o).all(), "Class sets differ!"

    if metric.lower() == "recall":
        mb, mo, title = rec_b, rec_o, "Per-class Recall"
    elif metric.lower() == "precision":
        mb, mo, title = pre_b, pre_o, "Per-class Precision"
    else:
        mb, mo, title = f1_b, f1_o, "Per-class F1"

    x = np.arange(len(classes_b)); w = 0.38
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x - w/2, mb, width=w, label="Baseline", alpha=0.85)
    ax.bar(x + w/2, mo, width=w, label="Ours",     alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(classes_b, rotation=90)
    ax.set_ylim(0.0, 1.05); ax.set_ylabel(metric.capitalize())
    ax.set_title(f"{title}: Baseline vs Ours"); ax.legend()

    for c in highlight:
        if c in classes_b:
            i = int(np.where(classes_b == c)[0])
            ax.axvspan(i-0.5, i+0.5, color="#ffe6e6", zorder=-1)

    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close(fig)

def plot_embedding(feats, y, method="umap",
                   save_path="results/embed.png",
                   sample_per_class=800, seed=0):
    _ensure_dir(os.path.dirname(save_path) or ".")
    rng = np.random.default_rng(seed)

    classes = np.unique(y)
    idx_sel = []
    for c in classes:
        idx_c = np.where(y == c)[0]
        idx_sel.append(rng.choice(idx_c, size=min(sample_per_class, len(idx_c)), replace=False)
                       if len(idx_c) > sample_per_class else idx_c)
    idx_sel = np.concatenate(idx_sel, axis=0)

    X = feats[idx_sel]; Y = y[idx_sel]

    Z = None
    if method.lower() == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=seed)
            Z = reducer.fit_transform(X)
        except Exception:
            method = "tsne"
    if method.lower() == "tsne":
       
        Z = TSNE(n_components=2, learning_rate="auto", init="random",
                 perplexity=35, random_state=seed).fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=Y, s=6, cmap="tab20")
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, ticks=classes)
    ax.set_title(f"{method.upper()} embedding")
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close(fig)

def plot_center_similarity(centers, save_path="results/center_similarity.png",
                           title="Center cosine similarity"):
    _ensure_dir(os.path.dirname(save_path) or ".")
    Z = centers / np.clip(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12, None)
    S = Z @ Z.T
    np.fill_diagonal(S, 0.0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(S, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title); ax.set_xlabel("Class"); ax.set_ylabel("Class")
    ax.set_xticks(np.arange(S.shape[0])); ax.set_yticks(np.arange(S.shape[0]))
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close(fig)

def plot_inter_intra_distributions(feats, y, centers,
                                   save_path="results/inter_intra_distributions.png"):
    _ensure_dir(os.path.dirname(save_path) or ".")
    f = feats   / np.clip(np.linalg.norm(feats,   axis=1, keepdims=True), 1e-12, None)
    c = centers / np.clip(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12, None)

    sim = f @ c.T
    own_sim = sim[np.arange(len(f)), y.astype(int)]
    own_dist = 1.0 - own_sim
    sim[np.arange(len(f)), y.astype(int)] = -np.inf
    nearest_other_sim = sim.max(axis=1)
    inter_dist = 1.0 - nearest_other_sim

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([own_dist, inter_dist],
               labels=["Intra (to own center)", "Inter (to nearest other)"])
    ax.set_ylabel("Cosine distance")
    ax.set_title("Intra vs. Inter-class distance distributions")
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close(fig)


def plot_ce_loss(history, save_path="results/ce_loss.png"):
    """
    Single-panel plot for cross-entropy (train) only.
    Needs: history['epoch'], history['train_ce']
    """
    _ensure_dir(os.path.dirname(save_path) or ".")
    ep = history.get("epoch", list(range(1, len(history.get("train_ce", [])) + 1)))

    fig, ax = plt.subplots(figsize=(8,4))
    if "train_ce" in history and len(history["train_ce"]):
        ax.plot(ep, history["train_ce"], label="Train CE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CE Loss")
    ax.set_title("Cross-Entropy Loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_validation_metrics(history, save_path="results/val_metrics.png"):
    """
    Validation metrics only (no lambda, no SupCon).
    Plots any of the available keys: val_acc, val_bal_acc, val_macro_f1.
    """
    _ensure_dir(os.path.dirname(save_path) or ".")
    L = max(
        len(history.get("val_acc", [])),
        len(history.get("val_bal_acc", [])),
        len(history.get("val_macro_f1", [])),
        1
    )
    ep = history.get("epoch", list(range(1, L + 1)))

    fig, ax = plt.subplots(figsize=(9,4.5))
    if "val_macro_f1" in history and len(history["val_macro_f1"]):
        ax.plot(ep, history["val_macro_f1"], label="Val Macro-F1")
    if "val_bal_acc" in history and len(history["val_bal_acc"]):
        ax.plot(ep, history["val_bal_acc"], label="Val Balanced Acc.")
    if "val_acc" in history and len(history["val_acc"]):
        ax.plot(ep, history["val_acc"], label="Val Acc")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric")
    ax.set_title("Validation Metrics")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def summarize_metrics(y_true, y_pred):
    return {
        "acc": accuracy_score(y_true, y_pred),
        "bal_acc": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


def plot_real_vs_diffusion_counts(train_counts, synth_counts,
                                  save_path="results/diffusion_balance_counts.png"):
    """
    Side-by-side bars of real (raw) vs. diffusion-synthetic counts per class.
    'After' = real + synthetic. Highlights how diffusion equalizes classes.
    """
    _ensure_dir(os.path.dirname(save_path) or ".")
    import numpy as np

    C = max(len(train_counts), len(synth_counts))
    real = np.array(train_counts, dtype=float)
    synth = np.array(synth_counts, dtype=float)
    if len(real) < C:  real = np.pad(real, (0, C - len(real)))
    if len(synth) < C: synth = np.pad(synth, (0, C - len(synth)))
    after = real + synth

    x = np.arange(C)
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x - w/2, real,  width=w, label="Real (pre-diffusion)", alpha=0.85)
    ax.bar(x + w/2, after, width=w, label="After (real + diffusion)", alpha=0.85)
    ax.set_xlabel("Class")
    ax.set_ylabel("Counts")
    ax.set_xticks(x)
    ax.set_title("Per-class counts: before vs. after diffusion balancing")
    ax.legend()
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close(fig)


def plot_effective_training_distribution(train_counts, synth_counts,
                                         save_path="results/diffusion_effective_distribution.png"):
    """
    Stacked bars per class: fraction synthetic vs real in the effective training mix.
    Also overlays the PRE vs POST normalized distributions for comparison.
    """
    _ensure_dir(os.path.dirname(save_path) or ".")
    import numpy as np

    real = np.array(train_counts, dtype=float)
    synth = np.array(synth_counts, dtype=float)
    C = max(len(real), len(synth))
    if len(real) < C:  real = np.pad(real, (0, C - len(real)))
    if len(synth) < C: synth = np.pad(synth, (0, C - len(synth)))
    post = real + synth

    pre_pdf  = real / max(1.0, real.sum())
    post_pdf = post / max(1.0, post.sum())

    x = np.arange(C)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(x, real,  label="Real", bottom=0.0, alpha=0.85)
    ax.bar(x, synth, label="Synthetic (diffusion)", bottom=real, alpha=0.85)

    ax2 = ax.twinx()
    ax2.plot(x, pre_pdf,  marker="o", linewidth=1.75, label="Pre (normalized)")
    ax2.plot(x, post_pdf, marker="s", linewidth=1.75, label="Post (normalized)")

    ax.set_xlabel("Class"); ax.set_ylabel("Counts")
    ax2.set_ylabel("Fraction")
    ax.set_title("Effective training mix: real vs diffusion per class\n(+ normalized pre/post overlays)")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close(fig)

def _filter_by_nn_distance(X_src, X_ref, remove_frac: float = 0.15):

    import numpy as np
    if (remove_frac is None) or (remove_frac <= 0) or (X_src.size == 0) or (X_ref.size == 0):
        return X_src
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(X_ref)
    dists, _ = nn.kneighbors(X_src, n_neighbors=1, return_distance=True)
    dists = dists.reshape(-1)
    thr = np.quantile(dists, remove_frac)
    keep = dists >= thr
    return X_src[keep] if keep.any() else X_src

def plot_tsne_triplet(
    feats, y, gen, fault_id,
    save_path="results/embed_fault_triplet.png",
    normalize="both",
    max_per_class=600,   # used as max faults
    ratio=5,
    seed=0,
    perplexity=35,
    pca_dim=50,
    align_gen=True,
    shrink=0.55,
    clip_q=0.94,
    remove_frac_norm=0.20,
    remove_frac_gen=0.20,
):

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)

    idx_norm = np.where(y == 0)[0]
    idx_fk   = np.where(y == int(fault_id))[0]

    Gk = np.asarray(gen) if gen is not None else np.empty((0, feats.shape[1]))
    if len(idx_norm) == 0 or len(idx_fk) == 0 or Gk.shape[0] == 0:
        raise ValueError("Missing data for plot_tsne_triplet (normal/fault/gen).")

    # ratio control: normal = ratio * F, generated = (ratio-1) * F
    ratio = max(2, int(ratio))

    max_fault_avail = min(len(idx_fk), max_per_class)
    max_norm_avail = len(idx_norm)
    max_gen_avail = Gk.shape[0]

    # constraints so that we can satisfy the counts exactly
    F_max_by_norm = max_norm_avail // ratio
    F_max_by_gen = max_gen_avail // (ratio - 1)

    F_use = min(max_fault_avail, F_max_by_norm, F_max_by_gen)
    if F_use <= 0:
        raise ValueError("Not enough samples to satisfy requested ratio in plot_tsne_triplet.")

    # sample indices
    idx_fk = rng.choice(idx_fk, size=F_use, replace=False)
    N_norm = ratio * F_use
    N_gen = (ratio - 1) * F_use

    idx_norm = rng.choice(idx_norm, size=N_norm, replace=False)
    if Gk.shape[0] > N_gen:
        Gk = Gk[rng.choice(np.arange(Gk.shape[0]), size=N_gen, replace=False)]

    Xn = feats[idx_norm]
    Xk = feats[idx_fk]
    # Gk already defined


    def l2n(a, eps=1e-12):
        n = np.linalg.norm(a, axis=1, keepdims=True)
        return a / np.clip(n, eps, None)

    if normalize.lower() in ("unit", "both"):
        Xn = l2n(Xn); Xk = l2n(Xk); Gk = l2n(Gk)
    if normalize.lower() in ("zscore", "both"):
        ref = np.vstack([Xn, Xk])
        scaler = StandardScaler().fit(ref)
        Xn = scaler.transform(Xn); Xk = scaler.transform(Xk); Gk = scaler.transform(Gk)

    def cov_sqrtm(C):
        w, V = np.linalg.eigh(C)
        w = np.clip(w, 1e-8, None)
        return V @ (np.sqrt(w)[:, None] * V.T), V @ ((1.0/np.sqrt(w))[:, None] * V.T)

    def mahal_clip(X, mu, C, q=0.97):
        Xm = X - mu
        w, V = np.linalg.eigh(C)
        w = np.clip(w, 1e-8, None)
        Z = Xm @ V
        Z = Z / np.sqrt(w)
        d2 = np.sum(Z*Z, axis=1)
        from scipy.stats import chi2
        thr = chi2.ppf(q, df=X.shape[1])
        keep = d2 <= thr
        return X[keep] if keep.any() else X

    if align_gen and Gk.size and Xk.size:
        mu_f = Xk.mean(axis=0, keepdims=True)
        mu_g = Gk.mean(axis=0, keepdims=True)
        Cf = np.cov(Xk.T); Cg = np.cov(Gk.T); I = np.eye(Cf.shape[0], dtype=Cf.dtype)
        Cf_s = (1.0 - shrink) * Cf + shrink * np.trace(Cf)/Cf.shape[0] * I
        Cg_s = (1.0 - shrink) * Cg + shrink * np.trace(Cg)/Cg.shape[0] * I
        Af, Afi = cov_sqrtm(Cf_s); Ag, Agi = cov_sqrtm(Cg_s)
        Gk = (Gk - mu_g) @ Agi @ Af + mu_f
        if clip_q is not None:
            Xk = mahal_clip(Xk, mu_f, Cf_s, q=clip_q)
            Gk = mahal_clip(Gk, mu_f, Cf_s, q=clip_q)

    Xfg = np.vstack([Xk, Gk]) if Gk.size else Xk
    Xn = _filter_by_nn_distance(Xn, Xfg, remove_frac=remove_frac_norm)
   
    if Gk.size:
        Gk = _filter_by_nn_distance(Gk, Xn, remove_frac=remove_frac_gen)

    # TSNE
    X_all = np.vstack([Xn, Xk, Gk])
    y_all = np.hstack([np.zeros(len(Xn), int), np.ones(len(Xk), int), np.full(len(Gk), 2, int)])

    if pca_dim is not None and pca_dim > 0 and X_all.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=seed)
        X_all = pca.fit_transform(X_all)

    n_total = X_all.shape[0]
    perp = int(max(5, min(perplexity, (n_total - 1) // 3))) if n_total > 10 else 5

    Z = TSNE(
        n_components=2, metric="euclidean", init="pca",
        perplexity=perp, early_exaggeration=16.0, learning_rate=100,
        random_state=seed,
    ).fit_transform(X_all)

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    ax.scatter(Z[y_all==0,0], Z[y_all==0,1], s=18, label="Normal", alpha=0.9)
    ax.scatter(Z[y_all==1,0], Z[y_all==1,1], s=26, label=f"Fault {fault_id}", alpha=0.9)
    ax.scatter(Z[y_all==2,0], Z[y_all==2,1], s=22, label=f"Fault {fault_id} Generated", alpha=0.9)
    ax.set_title("t-SNE: Normal vs Fault vs Generated")
    ax.legend()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close(fig)

def plot_tsne_normal_fault6_generated(
    feats_real,
    y_real,
    gen_fault6_feats=None,
    normal_label=0,
    fault6_label=6,
    save_path="results/tsne_norm_f6_gen.png",
    max_per_group=650,
    seed=0,
    normalize="both",
    normal_cap=350,
    fault_real_cap=40,
    perplexity=35,
    pca_dim=50,
    align_gen=True,
    shrink=0.55,
    clip_q=0.94,
    remove_frac_norm=0.20,   
    remove_frac_gen=0.20,    
):
    import os
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)

    idx_norm = np.where(y_real == int(normal_label))[0]
    idx_fault = np.where(y_real == int(fault6_label))[0]
    if len(idx_norm) == 0 or len(idx_fault) == 0:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig, ax = plt.subplots(figsize=(7.6, 5.6))
        ax.text(0.5, 0.5, "no data for requested classes", ha="center", va="center")
        plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close(fig); return

    if len(idx_norm) > normal_cap:
        idx_norm = rng.choice(idx_norm, size=normal_cap, replace=False)
    if len(idx_fault) > fault_real_cap:
        idx_fault = rng.choice(idx_fault, size=fault_real_cap, replace=False)

    Xn = feats_real[idx_norm]
    Xf = feats_real[idx_fault]
    G  = np.asarray(gen_fault6_feats) if gen_fault6_feats is not None else np.empty((0, Xn.shape[1]))
    if G.shape[0] > max_per_group:
        G = G[rng.choice(np.arange(G.shape[0]), size=max_per_group, replace=False)]

    def l2n(a, eps=1e-12):
        n = np.linalg.norm(a, axis=1, keepdims=True)
        return a / np.clip(n, eps, None)

    if normalize.lower() in ("unit", "both"):
        if Xn.size: Xn = l2n(Xn)
        if Xf.size: Xf = l2n(Xf)
        if G.size:  G  = l2n(G)

    if normalize.lower() in ("zscore", "both"):
        ref = np.vstack([Xn, Xf]) if Xn.size and Xf.size else (Xn if Xn.size else Xf)
        scaler = StandardScaler().fit(ref)
        if Xn.size: Xn = scaler.transform(Xn)
        if Xf.size: Xf = scaler.transform(Xf)
        if G.size:  G  = scaler.transform(G)


    def cov_sqrtm(C):
        w, V = np.linalg.eigh(C)
        w = np.clip(w, 1e-8, None)
        return V @ (np.sqrt(w)[:, None] * V.T), V @ ((1.0/np.sqrt(w))[:, None] * V.T)

    def mahal_clip(X, mu, C, q=0.97):
        if X.size == 0: return X
        Xm = X - mu
        w, V = np.linalg.eigh(C)
        w = np.clip(w, 1e-8, None)
        Z = Xm @ V
        Z = Z / np.sqrt(w)
        d2 = np.sum(Z*Z, axis=1)
        from scipy.stats import chi2
        thr = chi2.ppf(q, df=X.shape[1])
        keep = d2 <= thr
        return X[keep] if keep.any() else X

    if align_gen and G.size and Xf.size:
        mu_f = Xf.mean(axis=0, keepdims=True)
        mu_g = G.mean(axis=0, keepdims=True)
        Cf = np.cov(Xf.T)
        Cg = np.cov(G.T)
        I = np.eye(Cf.shape[0], dtype=Cf.dtype)
        Cf_s = (1.0 - shrink) * Cf + shrink * np.trace(Cf)/Cf.shape[0] * I
        Cg_s = (1.0 - shrink) * Cg + shrink * np.trace(Cg)/Cg.shape[0] * I
        Af, Afi = cov_sqrtm(Cf_s)
        Ag, Agi = cov_sqrtm(Cg_s)
        G = (G - mu_g) @ Agi @ Af + mu_f
        if clip_q is not None:
            Xf = mahal_clip(Xf, mu_f, Cf_s, q=clip_q)
            G  = mahal_clip(G,  mu_f, Cf_s, q=clip_q)


    Xfg = Xf if G.size == 0 else np.vstack([Xf, G])
    Xn = _filter_by_nn_distance(Xn, Xfg, remove_frac=remove_frac_norm)
    if G.size:
        G = _filter_by_nn_distance(G, Xn, remove_frac=remove_frac_gen)

    # --- TSNE ---
    X_all = np.vstack([Xn, Xf, G]) if G.size else np.vstack([Xn, Xf])
    y_all = (np.hstack([np.zeros(len(Xn), int), np.ones(len(Xf), int), np.full(len(G), 2, int)])
             if G.size else np.hstack([np.zeros(len(Xn), int), np.ones(len(Xf), int)]))

    if pca_dim is not None and pca_dim > 0 and X_all.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=seed)
        X_all = pca.fit_transform(X_all)

    n_total = X_all.shape[0]
    perp = int(max(5, min(perplexity, (n_total - 1) // 3))) if n_total > 10 else 5

    Z = TSNE(
        n_components=2, metric="euclidean", init="pca",
        perplexity=perp, early_exaggeration=16.0, learning_rate=100,
        random_state=seed,
    ).fit_transform(X_all)

    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    ax.scatter(Z[y_all==0,0], Z[y_all==0,1], s=20, label="Normal")
    ax.scatter(Z[y_all==1,0], Z[y_all==1,1], s=28, label=f"Fault {fault6_label}")
    if (y_all == 2).any():
        ax.scatter(Z[y_all==2,0], Z[y_all==2,1], s=22, label=f"Fault {fault6_label} Generated")
    ax.legend(loc="best", frameon=True)
    ax.set_title("t-SNE: Normal vs Fault vs Generated")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close(fig)

def plot_confusion_matrix_heatmap(
    y_true,
    y_pred,
    num_classes,
    save_path="results/confusion_matrix_test.png",
):
    """
    Normalized confusion matrix heatmap that prints a value in every block.
    Values are shown with 2 decimals, including zeros (0.00).
    """

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # Row normalize
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm.astype(float) / np.clip(row_sums, 1e-12, None)

    fig, ax = plt.subplots(figsize=(12, 9))

    # Heatmap
    im = ax.imshow(
        cm_norm,
        interpolation="nearest",
        cmap="Greens",
        vmin=0.0,
        vmax=1.0,
    )

    # Axis labels
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    ax.set_title("Normalized Confusion Matrix")

    ax.tick_params(axis="both", which="major", labelsize=7)

    # Color bar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Probability")

    # Annotate each cell with value (2 decimals)
    fmt = ".2f"
    for i in range(num_classes):
        for j in range(num_classes):
            v = cm_norm[i, j]
            # text color contrast
            text_color = "white" if v > 0.5 else "black"
            ax.text(
                j,
                i,
                format(v, fmt),
                ha="center",
                va="center",
                color=text_color,
                fontsize=6,
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)



def gates_to_sensor_segment_matrix(extras, reduce="max"):
    """
    Convert per-d_model gates into per-sensor × segment weights.

    Approx attribution: weight[f, s] ≈ ∑_d |W_proj[f,d]| * gate[s,d]

    extras:
        "gates": (B, S, d_model) from model(..., return_gates=True)
        "W_proj": (d_model, F) input projection matrix
    """
    gates = extras["gates"]
    W_proj = extras["W_proj"]

    if isinstance(gates, torch.Tensor):
        G = gates.detach().cpu().numpy()
    else:
        G = np.asarray(gates)

    if reduce == "max":
        G = G.max(axis=0)      # (S, d_model)
    else:
        G = G.mean(axis=0)     # (S, d_model)

    if isinstance(W_proj, torch.Tensor):
        Wabs_T = torch.abs(W_proj).T.detach().cpu().numpy()  # (F, d_model)
    else:
        Wabs_T = np.abs(np.asarray(W_proj)).T

    # (F, d_model) @ (d_model, S) -> (F, S)
    M = Wabs_T @ G.T
    return M


def plot_topk_sensors(
    M,
    sensor_names=None,
    k=10,
    fault_id=None,
    out_png="results/gating_top10.png",
    label="Mean gate weight (across segments)",
    title_prefix="Top sensors by gate weight",
):
    """
    Bar plot of top-k sensors based on average weight across segments.

    M: (F, S) sensor × segment matrix
    """
    F, S = M.shape
    if sensor_names is None:
        sensor_names = [f"Var{i+1}" for i in range(F)]

    mean_per_sensor = M.mean(axis=1)  # (F,)
    idx = np.argsort(-mean_per_sensor)[:k]
    vals = mean_per_sensor[idx][::-1]
    names = [sensor_names[i] for i in idx][::-1]

    plt.figure(figsize=(8, max(4, 0.4 * k + 2)))
    plt.barh(np.arange(len(vals)), vals)
    plt.yticks(np.arange(len(vals)), names)
    plt.xlabel(label)
    title = title_prefix
    if fault_id is not None:
        title += f" - Fault {fault_id}"
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

