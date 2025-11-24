import yaml

def compute_train_ratio(cfg_path: str, num_fault_classes: int = 20):
    """
    Compute Normal vs ONE fault ratio for training data
    using only config.yaml (no dataset loading).
    """
    cfg = yaml.safe_load(open(cfg_path)) or {}
    dw = cfg.get("data_windowing", {}) or {}

    # Normal (fault 0) slice
    n_train_start = int(dw.get("normal_train_start", 0))
    n_train_end   = int(dw.get("normal_train_end", 42000))
    normal_train = max(0, n_train_end - n_train_start)

    # Fault slices
    post_fault_start   = int(dw.get("post_fault_start", 100))
    per_run_fault_len  = max(0, 660 - post_fault_start)

    tr_start = int(dw.get("train_runs_start", 1))
    tr_end   = int(dw.get("train_runs_end", 25))   # end-exclusive
    n_train_runs = max(0, tr_end - tr_start)

    per_fault_train = n_train_runs * per_run_fault_len

    # Ratio Normal : ONE Fault
    ratio = (normal_train / per_fault_train) if per_fault_train > 0 else float('inf')

    return {
        "normal_train": normal_train,
        "per_fault_train": per_fault_train,
        "train_normal_to_one_fault_ratio": ratio,
        "n_train_runs": n_train_runs,
        "per_run_fault_len": per_run_fault_len,
    }

def pretty_print_train(stats):
    """
    Pretty-print the training counts and ratios.
    """
    print("=== Train counts from config (no data read) ===")
    print(f"Train runs: {stats['n_train_runs']}")
    print(f"Per-run faulty slice length:        {stats['per_run_fault_len']} rows\n")
    print(f"Normal (train):                     {stats['normal_train']}")
    print(f"Per-fault (train):                  {stats['per_fault_train']}")
    print(f"Normal : ONE fault (train):         {stats['normal_train']} : {stats['per_fault_train']} "
          f"(~{stats['train_normal_to_one_fault_ratio']:.2f} : 1)")
