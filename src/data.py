import numpy as np
import pandas as pd
import pyreadr
from pathlib import Path

import yaml

try:
    import cupy as cp
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    GPU_AVAILABLE = True
except Exception:
    cp = np  # fallback
    GPU_AVAILABLE = False

from sklearn.preprocessing import StandardScaler as skStandardScaler


def _try_load_slices_from_config():
    """
    Try to load normal (fault 0) train/test slice bounds from config.yaml.

    Returns
    -------
    dict with keys:
        normal_train_start, normal_train_end, normal_test_start, normal_test_end
    or {} if config not found / keys not present.
    """
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    if not cfg_path.exists():
        return {}
    try:
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        dw = (cfg.get("data_windowing") or {})
        return {
            "normal_train_start": int(dw.get("normal_train_start"))
                if dw.get("normal_train_start") is not None else None,
            "normal_train_end": int(dw.get("normal_train_end"))
                if dw.get("normal_train_end") is not None else None,
            "normal_test_start": int(dw.get("normal_test_start"))
                if dw.get("normal_test_start") is not None else None,
            "normal_test_end": int(dw.get("normal_test_end"))
                if dw.get("normal_test_end") is not None else None,
        }
    except Exception:
        return {}


def read_training_data(ff_path: str, ft_path: str):
    ff = Path(ff_path)
    ft = Path(ft_path)
    if not ff.exists():
        raise FileNotFoundError(f"Fault-free RData not found: {ff}")
    if not ft.exists():
        raise FileNotFoundError(f"Faulty RData not found: {ft}")

    ff_r = pyreadr.read_r(str(ff))
    ft_r = pyreadr.read_r(str(ft))

    # Auto detect key names based on filename
    if "Testing" in ff.name:
        key_free = "fault_free_testing"
    else:
        key_free = "fault_free_training"

    if "Testing" in ft.name:
        key_faulty = "faulty_testing"
    else:
        key_faulty = "faulty_training"

    if key_free not in ff_r:
        raise KeyError(f"Key {key_free} not in {ff}")
    if key_faulty not in ft_r:
        raise KeyError(f"Key {key_faulty} not in {ft}")

    b1 = ff_r[key_free]
    b2 = ft_r[key_faulty]

    return pd.concat([b1, b2]).sort_values(["faultNumber", "simulationRun"])


def sample_train_and_test(
    train_ts,
    type_model: str = "supervised",
    post_fault_start: int = 100,
    train_runs=range(1, 25),
    test_runs=range(26, 38),
    normal_train_start: int = 0,
    normal_train_end: int = 42000,
    normal_test_start: int = 42000,
    normal_test_end: int = 44000,
):
    """
    Slice training/testing windows from the TEP table.

    Parameters
    ----------
    type_model : "supervised" or other
    post_fault_start : int
        index in each faulty run from which to take post-fault windows
    train_runs, test_runs : iterable of run ids
    normal_* : ints
        start/end indices for fault 0 (normal) rows to include in train/test
        (equivalent to fault_0.iloc[start:end])
    """
    frames_train, frames_test = [], []
    fault_0 = train_ts[train_ts["faultNumber"] == 0]

    # ---------------- TRAIN ----------------
    if type_model == "supervised":
        for i in sorted(train_ts["faultNumber"].unique()):
            if i == 0:
                frames_train.append(fault_0.iloc[normal_train_start:normal_train_end])
            else:
                b = train_ts[train_ts["faultNumber"] == i]
                per = []
                for x in train_runs:
                    bx = b[b["simulationRun"] == x]
                    if not bx.empty:
                        per.append(bx.iloc[20:500])
             
                if per:
                    frames_train.append(pd.concat(per))
    else:
        frames_train.append(fault_0.iloc[normal_train_start:normal_train_end])


    if not frames_train:
        frames_train.append(train_ts.iloc[0:0].copy())

    sampled_train = pd.concat(frames_train).sort_values(
        ["faultNumber", "simulationRun", "sample"]
    )

    # ---------------- TEST ----------------
    for i in sorted(train_ts["faultNumber"].unique()):
        if i == 0:
            frames_test.append(fault_0.iloc[normal_test_start:normal_test_end])
        else:
            b = train_ts[train_ts["faultNumber"] == i]
            per = []
            for x in test_runs:
                bx = b[b["simulationRun"] == x]
                if not bx.empty:
                    per.append(bx.iloc[post_fault_start:660])
            if per:
                frames_test.append(pd.concat(per))

    if not frames_test:
        frames_test.append(train_ts.iloc[0:0].copy())

    sampled_test = pd.concat(frames_test).sort_values(
        ["faultNumber", "simulationRun", "sample"]
    )
    return sampled_train.reset_index(drop=True), sampled_test.reset_index(drop=True)



def scale_and_window(X_df, scaler, use_gpu=True, y_col='faultNumber',
                     window_size=20, stride=5, return_meta=True):
    """
    Scale + window the multivariate time series into (N_windows, T, F).
    """
    y = X_df[y_col].values
    run_ids = X_df['simulationRun'].values
    samples = X_df['sample'].values
    X = X_df.iloc[:, 3:].values  # features only

    if use_gpu and GPU_AVAILABLE:
        X_scaled = scaler.transform(cp.asarray(X))
        n = max(0, (len(X_scaled) - window_size) // stride + 1)
        X_idx = cp.arange(window_size)[None, :] + cp.arange(n)[:, None] * stride
        y_idx = cp.arange(window_size - 1, window_size - 1 + n * stride, stride)
        X_win = X_scaled[X_idx].get()
        y_win = cp.asarray(y)[y_idx].get()
        run_win = cp.asarray(run_ids)[y_idx].get()
        samp_win = cp.asarray(samples)[y_idx].get()
    else:
        X_scaled = scaler.transform(X)
        n = max(0, (len(X_scaled) - window_size) // stride + 1)
        X_idx = np.arange(window_size)[None, :] + np.arange(n)[:, None] * stride
        X_win = np.stack([X_scaled[idx] for idx in X_idx], axis=0)
        y_idx = np.arange(window_size - 1, window_size - 1 + n * stride, stride)
        y_win = y[y_idx]
        run_win = run_ids[y_idx]
        samp_win = samples[y_idx]

    if return_meta:
        meta = {'faultNumber': y_win, 'simulationRun': run_win, 'sample_end': samp_win}
        return X_win, y_win, meta
    return X_win, y_win


def load_sampled_data(*,
                      window_size=100,
                      stride=5,
                      type_model="supervised",
                      use_gpu=True,
                      ff_path: str,
                      ft_path: str,
                      post_fault_start=100,
                      train_runs=range(1, 25),
                      test_runs=range(26, 38),
                      normal_train_start: int = None,
                      normal_train_end: int = None,
                      normal_test_start: int = None,
                      normal_test_end: int = None):
    """
    End to end loading + scaling + windowing.
    """
    cfg_slices = _try_load_slices_from_config()
    ntr_s = normal_train_start if normal_train_start is not None else cfg_slices.get("normal_train_start", 0)
    ntr_e = normal_train_end   if normal_train_end   is not None else cfg_slices.get("normal_train_end",   42000)
    nte_s = normal_test_start  if normal_test_start  is not None else cfg_slices.get("normal_test_start",  42000)
    nte_e = normal_test_end    if normal_test_end    is not None else cfg_slices.get("normal_test_end",    44000)


    ts = read_training_data(ff_path, ft_path)

    tr, te = sample_train_and_test(
        ts,
        type_model,
        post_fault_start=post_fault_start,
        train_runs=train_runs,
        test_runs=test_runs,
        normal_train_start=ntr_s,
        normal_train_end=ntr_e,
        normal_test_start=nte_s,
        normal_test_end=nte_e,
    )

    fault_free = tr[tr["faultNumber"] == 0].iloc[:, 3:].values
    if use_gpu and GPU_AVAILABLE:
        scaler = cuStandardScaler()
        scaler.fit(cp.asarray(fault_free))
    else:
        scaler = skStandardScaler()
        scaler.fit(fault_free)

    X_train, y_train, meta_tr = scale_and_window(
        tr,
        scaler,
        use_gpu=use_gpu,
        y_col="faultNumber",
        window_size=window_size,
        stride=stride,
        return_meta=True,
    )
    X_test, y_test, meta_te = scale_and_window(
        te,
        scaler,
        use_gpu=use_gpu,
        y_col="faultNumber",
        window_size=window_size,
        stride=stride,
        return_meta=True,
    )
    return (X_train, y_train, meta_tr), (X_test, y_test, meta_te)
