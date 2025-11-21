import os
import json
import warnings

warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", message=".*enable_nested_tensor is True.*")

import argparse
from pathlib import Path
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from src.diffusion_trainer import train_decision_diffusion_streaming
from collections import defaultdict

try:
    from imblearn.over_sampling import SMOTE
    IMB_AVAILABLE = True
except Exception:
    IMB_AVAILABLE = False

from .data import load_sampled_data
from .datasets import (TwoCropsTransform, TSJitter, TSScale, TSTimeMask,
                       ContrastiveTSDataset, PlainTSDataset, make_sampler)
from .model import SelfGatedHierarchicalTransformerEncoder, CosineMarginClassifier
from .proto import PrototypeCenters, CenterSeparationLoss
from .losses import train_one_epoch
from .metrics import evaluate
from src.plots import get_preds_and_logits, get_features


def pretty_print_metrics(tag, m):
    acc = m.get("acc", None)
    bal = m.get("bal_acc", None)
    f1  = m.get("macro_f1", None)
    print(f"{tag}: Acc={acc:.3f} | BalAcc={float(bal):.3f} | MacroF1={f1:.3f}")


def parse_args():
    p = argparse.ArgumentParser(description="TEP classifier training (config-driven)")
    p.add_argument("--config", type=str, default=str(Path(__file__).parent.parent / "config.yaml"),
                   help="Path to config.yaml")
                   
    p.add_argument("--ff-path", type=str, default=None)
    p.add_argument("--ft-path", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--train-batch", type=int, default=None)
    p.add_argument("--val-batch", type=int, default=None)
    p.add_argument("--test-batch", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--window-size", type=int, default=None)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--post-fault-start", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--results-dir", type=str, default=None,
               help="Where to save artifacts (overrides YAML)")

    p.add_argument("--baseline", action="store_true",
                help="Use only the linear classifier (no cosine head / contrastive / centers)")
    p.add_argument("--arcface-only", action="store_true",
               help="Use cosine-margin head with CE only (disables SupCon and center losses)")
    p.add_argument("--smote", action="store_true",
                   help="Apply SMOTE to training windows after split (avoid using WeightedRandomSampler then)")
    p.add_argument("--smote-ratio", type=float, default=None,
                   help="Target per-class size as a fraction of the majority (e.g., 0.5). If omitted, upsample minorities to majority.")

    return p.parse_args()

def set_seed(seed: int):
    import os
    import random
    import numpy as np
    import torch

    # Python
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # Torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    import numpy as np
    import torch
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

# ------------------------
# Plain CE training (baseline)
# ------------------------
def _to_jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return x

def train_one_epoch_ce(model, loader, opt, device):
    model.train()
    tot, tot_loss = 0, 0.0
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            x, y = batch[0], batch[1]
        else:
            x, y = batch
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        bs = y.size(0); tot += bs; tot_loss += loss.item() * bs
    return tot_loss / max(1, tot)


def main():
    args = parse_args()
    if args.baseline and args.arcface_only:
        raise ValueError("Choose either --baseline or --arcface-only, not both.")

    # ---- Load YAML config
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    results_dir = args.results_dir or cfg.get("training", {}).get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)

    seed = args.seed or cfg["training"]["seed"]
    set_seed(seed)

    ff_path = args.ff_path or cfg["dataset"]["ff_path"]
    ft_path = args.ft_path or cfg["dataset"]["ft_path"]

    window_size = args.window_size or cfg["data_windowing"]["window_size"]
    stride      = args.stride or cfg["data_windowing"]["stride"]
    post_fault_start = args.post_fault_start or cfg["data_windowing"]["post_fault_start"]

    train_runs = range(cfg["data_windowing"]["train_runs_start"], cfg["data_windowing"]["train_runs_end"])
    test_runs  = range(cfg["data_windowing"]["test_runs_start"],  cfg["data_windowing"]["test_runs_end"])

    epochs      = args.epochs or cfg["training"]["epochs"]
    train_bs    = args.train_batch or cfg["training"]["batch"]["train"]
    val_bs      = args.val_batch or cfg["training"]["batch"]["val"]
    test_bs     = args.test_batch or cfg["training"]["batch"]["test"]
    num_workers = args.num_workers or cfg["training"]["num_workers"]

    # ---- Data load
    (X_train, y_train, _), (X_test, y_test, _) = load_sampled_data(
        window_size=window_size, stride=stride, ff_path=ff_path, ft_path=ft_path,
        post_fault_start=post_fault_start, train_runs=train_runs, test_runs=test_runs
    )

    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    (train_idx, val_idx), = sss.split(X_train, y_train)
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    use_weighted_sampler = True
    if args.smote:
        if not IMB_AVAILABLE:
            raise ImportError("imblearn not available. Install with: pip install imbalanced-learn")

        counts = np.bincount(y_tr.astype(int))
        majority = counts.max()
        uniq = np.unique(y_tr)

        if args.smote_ratio is None:
            sampling_strategy = "not majority"  
            target_descr = f"to majority ({majority})"
        else:
            target = max(1, int(round(majority * float(args.smote_ratio))))
            sampling_strategy = {int(c): target for c in uniq if counts[int(c)] < target}
            target_descr = f"to ratio {args.smote_ratio} × majority (~{target})"

        min_minority = counts[counts > 0].min()
        k_neighbors = max(1, min(5, int(min_minority) - 1))
        print(f"[SMOTE] Upsampling minorities {target_descr}; k_neighbors={k_neighbors}")

        N, T, D = X_tr.shape
        X2d = X_tr.reshape(N, T * D).astype(np.float32)
        sm = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=seed
        )
        X_res, y_res = sm.fit_resample(X2d, y_tr)
        X_tr, y_tr = X_res.reshape(-1, T, D), y_res



        use_weighted_sampler = False 

    # ---- Datasets
    use_cuda = torch.cuda.is_available()
    use_pin_memory = use_cuda

    train_ds = PlainTSDataset(X_tr, y_tr)


    # generator for reproducible DataLoader shuffling and sampling
    g = torch.Generator()
    g.manual_seed(seed)

    if use_weighted_sampler:
        sampler, class_counts = make_sampler(y_tr)
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=train_bs,
            sampler=sampler,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            persistent_workers=True,
            prefetch_factor=2,
            worker_init_fn=seed_worker,
            generator=g,
        )
    else:
        class_counts = np.bincount(y_tr.astype(int))
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=train_bs,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            persistent_workers=True,
            prefetch_factor=2,
            worker_init_fn=seed_worker,
            generator=g,
        )


    val_loader = torch.utils.data.DataLoader(
        PlainTSDataset(X_val, y_val), batch_size=val_bs, shuffle=False,
        num_workers=0, pin_memory=use_pin_memory
    )
    test_loader = torch.utils.data.DataLoader(
        PlainTSDataset(X_test, y_test), batch_size=test_bs, shuffle=False,
        num_workers=0, pin_memory=use_pin_memory
    )

    # ---- Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Model
    model = SelfGatedHierarchicalTransformerEncoder(
            input_dim=X_train.shape[2],
            d_model=128,
            nhead=4,
            num_layers_low=2,
            num_layers_high=2,
            dim_feedforward=128,
            dropout=0.05,
            pool_output_size=10,
            num_classes=int(y_train.max()) + 1,
        ).to(device)


    if args.baseline:
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        diffusion_model = None
        use_diffusion = False
    else:
        with torch.no_grad():
            first_batch = next(iter(train_loader))
            x0 = first_batch[0] if isinstance(first_batch, (list, tuple)) else first_batch
            x0 = x0.to(device)
            feat_dim = model.forward_features(x0).shape[-1]

        num_classes = model.classifier[-1].out_features
        model.cos_head = CosineMarginClassifier(
            feat_dim=feat_dim, num_classes=num_classes,
            s=float(cfg["model"]["s"]), m=float(cfg["model"]["m"]),
            margin_type=str(cfg["model"]["margin_type"])
        ).to(device)

        
        per_m = torch.full((num_classes,), float(cfg["model"]["m"]), device=device)
        for k, v in (cfg.get("model", {}).get("per_class_margin_overrides", {}) or {}).items():
            idx = int(k)
            if idx < num_classes:
                per_m[idx] = float(v)
        model.cos_head.per_class_margin = per_m

        centers = center_sep = None
        per_class_center_w = None
        if not args.arcface_only:
            centers = PrototypeCenters(
                num_classes=num_classes, feat_dim=feat_dim,
                momentum=float(cfg["model"]["center_momentum"]), device=device
            ).to(device)

            per_class_center_w = torch.ones(num_classes, device=device)
            for k, v in (cfg.get("model", {}).get("center_pull_weights", {}) or {}).items():
                idx = int(k)
                if idx < num_classes:
                    per_class_center_w[idx] = float(v)

            center_sep = CenterSeparationLoss(
                K=int(cfg["model"]["center_sep_K"]),
                margin=float(cfg["model"]["center_sep_margin"])
            ).to(device)

        opt = torch.optim.AdamW([
            {"params": [p for n, p in model.named_parameters() if not n.startswith("cos_head.")],
            "lr": 3e-4, "weight_decay": 1e-4},
            {"params": [model.cos_head.W], "lr": 3e-4, "weight_decay": 1e-4},
        ], lr=3e-4, weight_decay=0.0)
    
        # === Diffusion config ===
        diff_cfg = (cfg.get("training", {}).get("diffusion", {}) or {})
        use_diffusion = bool(diff_cfg.get("enabled", False))
        start_ep = int(diff_cfg.get("start_epoch", 3))
        synth_ratio = float(diff_cfg.get("synth_ratio", 0.20))
        diff_epochs = int(diff_cfg.get("epochs", 5))
        diff_steps_infer = int(diff_cfg.get("steps_infer", 24))
        diff_width = int(diff_cfg.get("width", 512))
        diff_depth = int(diff_cfg.get("depth", 3))
        margin_gate_delta = float(diff_cfg.get("margin_gate_delta", 0.05))
        autobalance = bool(diff_cfg.get("autobalance", True)) 

        diffusion_model = None   


    gen_counter = defaultdict(int)                 
    aug_stats = {"per_epoch_synth_counts": [],     
                 "num_classes": int(y_train.max()) + 1}

    train_counts = np.bincount(y_train.astype(int), minlength=int(y_train.max()) + 1)
    majority = int(train_counts.max())
    quota = {int(c): int(max(0, majority - cnt)) for c, cnt in enumerate(train_counts.tolist())}

    # ---- Train
    sampler_for_train = None

    best_bal_acc, best_state = 0.0, None
    history = {
        "epoch": [],
        "train_ce": [],
        "train_supcon": [],
        "lambda_supcon": [],
        "val_acc": [],
        "val_bal_acc": [],
        "val_macro_f1": []
    }
    
    for epoch in range(1, epochs + 1):

        if (not args.baseline) and use_diffusion and (epoch == start_ep):
            print("[Diffusion] Training decision-space diffusion (streaming)...")
            
            model.eval()

            feat_loader = torch.utils.data.DataLoader(
                PlainTSDataset(X_tr, y_tr),
                batch_size=val_bs,
                shuffle=False,
                num_workers=0,           
                pin_memory=True
            )

            diffusion_model = train_decision_diffusion_streaming(
                model=model,
                feat_loader=feat_loader,
                device=device,
                num_classes=int(y_train.max()) + 1,
                feat_dim=None,
                epochs=diff_epochs,
                bs=1024,
                lr=1e-3,
                T=1000,
                steps_infer=diff_steps_infer,
                width=diff_width,
                depth=diff_depth,
                use_project=False,        
                amp_enabled=True,
                microbatch=256,
                log_every=200,
                gen_counter=gen_counter,
                quota=quota,
                auto_balance_to_majority=autobalance,   
            )
            torch.save(
                diffusion_model.state_dict(),
                os.path.join(results_dir, "diffusion_state_dict.pt")
            )
            sampler_for_train = getattr(diffusion_model, "ddim_sample_raw", diffusion_model.ddim_sample)

            model.train()

        if args.baseline:
            ce = train_one_epoch_ce(model, train_loader, opt, device)
            con, lam = 0.0, 0.0
        else:
            # Margin warmup for arcface/cos head
            if hasattr(model, "cos_head"):
                T_half = max(1, int(0.3 * epochs))
                warm = min(1.0, (epoch / T_half) ** 2)
                num_classes = model.classifier[-1].out_features
                base_m = float(cfg["model"]["m"])
                per_m = torch.full((num_classes,), base_m, device=device)
                for k, v in (cfg.get("model", {}).get("per_class_margin_overrides", {}) or {}).items():
                    idx = int(k)
                    if idx < num_classes:
                        per_m[idx] = float(v)
                model.cos_head.per_class_margin = per_m * warm
                if epoch == 1:
                    print("[ARC] margin warmup enabled; targets:", per_m.tolist())

            if getattr(args, "arcface_only", False):
                ce, con, lam = train_one_epoch(
                                model, train_loader, opt, device, class_counts,
                                base_lambda=0.5, epoch=epoch, total_epochs=epochs,
                                mixup_alpha=0.4, mixup_prob=0.35,
                                centers=centers, per_class_center_w=per_class_center_w, lambda_center=0.010,
                                center_sep=center_sep, lambda_center_sep=0.010, temperature=0.12,
                                diffusion_sampler=sampler_for_train,        # <-- use RAW sampler here
                                synth_ratio=(synth_ratio if (not args.baseline) and use_diffusion and (diffusion_model is not None) and (epoch >= start_ep) else 0.0),
                                margin_gate_delta=margin_gate_delta
                            )

            else:
                ce, con, lam = train_one_epoch(
                        model, train_loader, opt, device, class_counts,
                        base_lambda=0.5, epoch=epoch, total_epochs=epochs,
                        mixup_alpha=0.4, mixup_prob=0.35,
                        centers=centers, per_class_center_w=per_class_center_w, lambda_center=0.010,
                        center_sep=center_sep, lambda_center_sep=0.010, temperature=0.12,
                        diffusion_sampler=sampler_for_train,
                        synth_ratio=(synth_ratio if (not args.baseline) and use_diffusion and (diffusion_model is not None) and (epoch >= start_ep) else 0.0),
                        margin_gate_delta=margin_gate_delta
                    )

        # Validation
        val = evaluate(model, val_loader, device)
        print(f"[{epoch:02d}] λ:{lam:.3f} CE:{ce:.4f} Con:{con:.4f} "
              f"Acc:{val['acc']:.3f} BalAcc:{val['bal_acc']:.3f} F1:{val['macro_f1']:.3f}")
        history["epoch"].append(epoch)
        history["train_ce"].append(ce)
        history["train_supcon"].append(con)
        history["lambda_supcon"].append(lam)
        history["val_acc"].append(val["acc"])
        history["val_bal_acc"].append(float(val["bal_acc"]))
        history["val_macro_f1"].append(val["macro_f1"])

        if val["bal_acc"] > best_bal_acc:
            best_bal_acc = val["bal_acc"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

  
        if (not args.baseline):
            C = aug_stats["num_classes"]
            snap = [0] * C
            for k, v in gen_counter.items():
                if 0 <= k < C:
                    snap[k] = int(v)
            aug_stats["per_epoch_synth_counts"].append(snap)
            gen_counter.clear()

    if best_state:
        model.load_state_dict(best_state)
        torch.save(best_state, os.path.join(results_dir, "best_state_dict.pt"))

     
    print("=== TEST (best regular) ===")
    test_m = evaluate(model, test_loader, device)
    pretty_print_metrics("TEST", test_m)

    with open(os.path.join(results_dir, "test_metrics.json"), "w") as f:
        json.dump(_to_jsonable(test_m), f, indent=2)

    np.savez(os.path.join(results_dir, "history.npz"), **history)

    try:
        y_true, y_pred, logits = get_preds_and_logits(model, test_loader, device)
        num_classes = int(np.max(y_true)) + 1

        # compute per class accuracy
        per_class_acc = {}
        for c in range(num_classes):
            mask = (y_true == c)
            if mask.sum() == 0:
                per_class_acc[c] = np.nan
            else:
                per_class_acc[c] = float((y_pred[mask] == y_true[mask]).mean())

        with open(os.path.join(results_dir, "per_class_acc.json"), "w") as f:
            json.dump({str(k): float(v) for k, v in per_class_acc.items()}, f, indent=2)

        with open(os.path.join(results_dir, "per_class_acc.csv"), "w") as f:
            f.write("class,acc,counts\n")
            for c in range(num_classes):
                n_c = int((y_true == c).sum())
                v = per_class_acc[c]
                acc_str = "" if not np.isfinite(v) else f"{100 * v:.4f}"
                f.write(f"{c},{acc_str},{n_c}\n")

        np.save(os.path.join(results_dir, "test_y.npy"), y_true)
        np.save(os.path.join(results_dir, "test_pred_reg.npy"), y_pred)
        np.save(os.path.join(results_dir, "test_logits_reg.npy"), logits)
    except Exception:
        pass



    try:
        np.save(os.path.join(results_dir, "y_train.npy"), y_train)
    except Exception:
        pass

    try:
        feats, y_all = get_features(model, test_loader, device)
        np.save(os.path.join(results_dir, "test_feats.npy"), feats)
        np.save(os.path.join(results_dir, "test_y.npy"), y_all)
    except Exception:
        pass

    try:
        if hasattr(model, "cos_head"):
            if 'centers' in locals() and hasattr(centers, "centers"):
                np.save(os.path.join(results_dir, "centers.npy"),
                        centers.centers.detach().cpu().numpy())
    except Exception:
        pass
    
    
    
    try:
        if (not args.baseline) and use_diffusion and (diffusion_model is not None):
            diff_cfg = (cfg.get("training", {}).get("diffusion", {}) or {})
            sel_faults = [int(x) for x in (diff_cfg.get("selected_faults", []) or [])]
            if not sel_faults:
                print("[EXPORT] No 'selected_faults' provided; skipping gated export.")
            else:
                export_per_class = int(diff_cfg.get("export_per_class", 300))
                export_steps = int(diff_cfg.get("export_steps_infer", diff_steps_infer))

                
                sampler = getattr(diffusion_model, "ddim_sample_raw", diffusion_model.ddim_sample)

             
                centers_np = None
                centers_path = os.path.join(results_dir, "centers.npy")
                if os.path.exists(centers_path):
                    centers_np = np.load(centers_path)

                def make_strict_gate(model, class_k: int, device="cuda",
                                    delta_logit=0.08,
                                    min_conf=0.55,
                                    normal_label=0,
                                    centers_tensor=None,
                                    min_cos_to_k=0.55,
                                    max_cos_to_normal=0.55):
                    # cos_head.W is (C, D) -> normalize rows and transpose to (D, C)
                    W = model.cos_head.W.to(device)                          # (C, D)
                    W = torch.nn.functional.normalize(W, dim=1)              # normalize each class vector
                    W_t = W.t()                                              # (D, C)
                    s = float(getattr(model.cos_head, "s", 30.0))

                    Ck = Cn = None
                    if centers_tensor is not None:
                        ct = torch.as_tensor(centers_tensor, dtype=torch.float32, device=device)  # (C, D)
                        ct = torch.nn.functional.normalize(ct, dim=-1)
                        if 0 <= class_k < ct.size(0):
                            Ck = ct[class_k]  # (D,)
                        if 0 <= normal_label < ct.size(0):
                            Cn = ct[normal_label]  # (D,)

                    def gate(z, y):
                        z = torch.nn.functional.normalize(z, dim=-1)         
                        logits = s * (z @ W_t)                               
                        probs  = torch.softmax(logits, dim=1)                
                        top2   = torch.topk(logits, 2, dim=1).values

                        is_topk   = (logits.argmax(1) == class_k)
                        conf_ok   = (probs[:, class_k] >= min_conf)
                        margin_ok = (top2[:, 0] - top2[:, 1] >=  delta_logit)

                        keep = is_topk & conf_ok & margin_ok
                        if Ck is not None:
                            keep = keep & ((z @ Ck) >= min_cos_to_k)
                        if Cn is not None:
                            keep = keep & ((z @ Cn) <= max_cos_to_normal)
                        return keep

                    return gate


                device = next(model.parameters()).device
                
                HARD_CLASSES = {15}           
                EXTRA_MAX_ROUNDS = 120        

                def export_with_gate(fid: int, n_keep: int, steps: int,
                                    chunk: int = 8192, max_rounds: int = 40):
                    """
                    Try increasingly lenient gates until we keep n_keep samples.
                    Special-case 'hard' classes with more leniency and more rounds.
                    """
                    local_max_rounds = EXTRA_MAX_ROUNDS if fid in HARD_CLASSES else max_rounds

                    def build_gate(delta_logit, min_conf, min_cos_to_k, max_cos_to_normal, use_centers=True):
                       
                        centers_arg = centers_np if (use_centers and centers_np is not None) else None
                        return make_strict_gate(
                            model, class_k=fid, device=device,
                            delta_logit=delta_logit,
                            min_conf=min_conf,
                            normal_label=0,
                            centers_tensor=centers_arg,
                            min_cos_to_k=min_cos_to_k,
                            max_cos_to_normal=max_cos_to_normal
                        )

                
                    settings = [
                        dict(delta_logit=0.15, min_conf=0.75, min_cos_to_k=0.80, max_cos_to_normal=0.30, use_centers=True),  # strict
                        dict(delta_logit=0.10, min_conf=0.60, min_cos_to_k=0.60, max_cos_to_normal=0.50, use_centers=True),  # medium
                        dict(delta_logit=0.06, min_conf=0.50, min_cos_to_k=0.45, max_cos_to_normal=0.65, use_centers=True),  # lenient+
                        dict(delta_logit=0.05, min_conf=0.45, min_cos_to_k=0.45, max_cos_to_normal=0.65, use_centers=True),

                  
                        dict(delta_logit=0.04, min_conf=0.42, min_cos_to_k=0.40, max_cos_to_normal=0.70, use_centers=False),
                        dict(delta_logit=0.03, min_conf=0.38, min_cos_to_k=0.35, max_cos_to_normal=0.75, use_centers=False),
                        dict(delta_logit=0.02, min_conf=0.32, min_cos_to_k=0.25, max_cos_to_normal=0.85, use_centers=False),

               
                        dict(delta_logit=0.00, min_conf=0.00, min_cos_to_k=-1.00, max_cos_to_normal=1.00, use_centers=False),
                    ]

                 
                    if fid in HARD_CLASSES:
                        settings = settings[2:]  

                    kept = None
                    for si, s in enumerate(settings, 1):
                        gate = build_gate(**s)
                        buf = []
                        rounds = 0
                        while sum(t.size(0) for t in buf) < n_keep and rounds < local_max_rounds:
                            rounds += 1
                            k = min(chunk, n_keep - sum(t.size(0) for t in buf))
                            y_prop = torch.full((k,), fid, dtype=torch.long, device=device)
                            Zprop  = sampler(y=y_prop, n=k, steps=steps, margin_gate=gate)
                            if Zprop is not None and Zprop.numel():
                                buf.append(Zprop.detach().cpu())

                        tot = sum(t.size(0) for t in buf)
                        if tot > 0:
                            kept = torch.cat(buf, dim=0)[:n_keep]
                            print(f"[EXPORT] fault {fid}: kept {kept.size(0)} / {n_keep} with gate setting #{si}")
                            break
                        else:
                            print(f"[EXPORT] fault {fid}: kept 0 with gate setting #{si}, relaxing...")

                    
                    if kept is None:
                        print(f"[EXPORT] fault {fid}: trying emergency fallback (no in-sampler gate).")
                        buf = []
                        rounds = 0
                        while sum(t.size(0) for t in buf) < n_keep and rounds < (local_max_rounds * 2):
                            rounds += 1
                            k = min(chunk, n_keep - sum(t.size(0) for t in buf))
                            y_prop = torch.full((k,), fid, dtype=torch.long, device=device)
                            Zprop  = sampler(y=y_prop, n=k, steps=steps, margin_gate=None)  # no gate
                            if Zprop is not None and Zprop.numel():
                               
                                W = F.normalize(model.cos_head.W.to(Zprop.device), dim=1).t()  # (D, C)
                                s = float(getattr(model.cos_head, "s", 30.0))
                                logits = s * (F.normalize(Zprop, dim=-1) @ W)
                                choose = (logits.argmax(1) == fid)
                                Zok = Zprop[choose]
                                if Zok.numel():
                                    buf.append(Zok.detach().cpu())

                        tot = sum(t.size(0) for t in buf)
                        if tot > 0:
                            kept = torch.cat(buf, dim=0)[:n_keep]
                            print(f"[EXPORT][fallback] fault {fid}: kept {kept.size(0)} / {n_keep}")
                        else:
                            print(f"[EXPORT] fault {fid}: FAILED to keep any samples after fallback.")
                            return torch.empty((0, diffusion_model.feat_dim), dtype=torch.float32)

                    return kept


                merged = {}
                for fid in sel_faults:
                    Zfid = export_with_gate(fid=fid, n_keep=export_per_class, steps=export_steps)
                    arr = Zfid.numpy()
                    
                    np.save(os.path.join(results_dir, f"gen_f{fid}.npy"), arr)
                    merged[str(fid)] = arr
                    print(f"[OK] Saved gated gen_f{fid}.npy (shape={arr.shape})")

              
                np.save(os.path.join(results_dir, "gen_selected.npy"), merged, allow_pickle=True)
                print(f"[OK] Saved gen_selected.npy for faults {sel_faults}")

          
                if hasattr(diffusion_model, "rescale_to_real_radii"):
                    merged_un = {}
                    for fid in sel_faults:
                        Zhat = torch.as_tensor(merged[str(fid)], device=device, dtype=torch.float32)
                        yhat = torch.full((Zhat.size(0),), fid, dtype=torch.long, device=device)
                        Zun  = diffusion_model.rescale_to_real_radii(Zhat, yhat).cpu().numpy()
                        np.save(os.path.join(results_dir, f"gen_f{fid}_unnorm.npy"), Zun)
                        merged_un[str(fid)] = Zun
                        print(f"[OK] Saved UNnormalized gen_f{fid}_unnorm.npy (shape={Zun.shape})")
                    np.save(os.path.join(results_dir, "gen_selected_unnorm.npy"), merged_un, allow_pickle=True)

    except Exception as e:
        print(f"[WARN] Could not export diffusion embeddings: {e}")




if __name__ == "__main__":
    main()
