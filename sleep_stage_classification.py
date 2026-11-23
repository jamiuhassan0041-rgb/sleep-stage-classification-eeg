"""Sleep stage classification from single-night EEG (Sleep-EDF Expanded, sleep-cassette).

Pipeline: download PSG + hypnogram -> band-pass filter -> 30 s epochs -> relative band power
(+ total power) on two EEG channels -> random forest -> leave-one-subject-out evaluation.

Run:  python sleep_stage_classification.py --subjects 0 1 2 3 4 5 6 7 8 9
Needs: pip install mne scikit-learn matplotlib pandas
Data:  Kemp et al. (2000), Sleep-EDF Expanded, PhysioNet (downloaded automatically by MNE).
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, cohen_kappa_score,
                             confusion_matrix, f1_score)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Stage descriptions in the hypnogram -> event codes 1..5 (stages 3 and 4 merged into N3).
STAGES = {"Sleep stage W": 1, "Sleep stage 1": 2, "Sleep stage 2": 3,
          "Sleep stage 3": 4, "Sleep stage 4": 4, "Sleep stage R": 5}
NAMES = ["Wake", "N1", "N2", "N3", "REM"]
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12), "sigma": (12, 16), "beta": (16, 30)}
CHANNELS = ["EEG Fpz-Cz", "EEG Pz-Oz"]
WAKE_MARGIN = 60  # keep 60 epochs (30 min) of wake before first and after last sleep epoch


def epochs_from_raw(raw):
    """Filter, cut into 30 s epochs, trim long wake periods. Returns Epochs or None."""
    raw = raw.copy().pick(CHANNELS)
    raw.filter(0.3, 35.0, verbose=False)
    events, _ = mne.events_from_annotations(raw, event_id=STAGES, chunk_duration=30.0, verbose=False)
    sleep = np.where(events[:, 2] != 1)[0]
    if len(sleep) == 0:
        return None
    lo, hi = max(0, sleep[0] - WAKE_MARGIN), min(len(events), sleep[-1] + WAKE_MARGIN + 1)
    events = events[lo:hi]
    tmax = 30.0 - 1.0 / raw.info["sfreq"]
    return mne.Epochs(raw, events, event_id={n: i + 1 for i, n in enumerate(NAMES)}, tmin=0.0,
                      tmax=tmax, baseline=None, preload=True, verbose=False)


def band_features(epochs):
    """Relative band power per channel and log10 total power per channel."""
    psd = epochs.compute_psd(method="welch", fmin=0.5, fmax=30.0,
                             n_fft=int(epochs.info["sfreq"] * 4), verbose=False)
    p, f = psd.get_data(return_freqs=True)            # (epochs, channels, freqs)
    total = p.sum(-1, keepdims=True)
    cols, feats = [], []
    for band, (lo, hi) in BANDS.items():
        feats.append(p[..., (f >= lo) & (f < hi)].sum(-1) / total[..., 0])
        cols += [f"{band}_{ch}" for ch in epochs.ch_names]
    feats.append(np.log10(total[..., 0]))
    cols += [f"logpower_{ch}" for ch in epochs.ch_names]
    return pd.DataFrame(np.concatenate(feats, axis=1), columns=cols)


def load_subject(idx, recording=1):
    from mne.datasets import sleep_physionet
    psg, hyp = sleep_physionet.age.fetch_data(subjects=[idx], recording=[recording])[0]
    raw = mne.io.read_raw_edf(psg, preload=True, verbose=False)
    raw.set_annotations(mne.read_annotations(hyp), emit_warning=False)
    return raw


def build_dataset(subjects, recording=1):
    frames = []
    for s in subjects:
        epochs = epochs_from_raw(load_subject(s, recording))
        if epochs is None:
            print(f"subject {s}: no sleep epochs, skipped")
            continue
        df = band_features(epochs)
        df["stage"] = epochs.events[:, 2] - 1
        df["subject"] = s
        frames.append(df)
        print(f"subject {s}: {len(df)} epochs")
    return pd.concat(frames, ignore_index=True)


def evaluate(df):
    """Leave-one-subject-out: every subject is predicted by a model that never saw them."""
    X, y, g = df.drop(columns=["stage", "subject"]), df["stage"].to_numpy(), df["subject"].to_numpy()
    pred = np.empty_like(y)
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        model = make_pipeline(StandardScaler(), RandomForestClassifier(
            n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=0))
        pred[te] = model.fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
    return y, pred, g


def report(y, pred, g, outdir):
    os.makedirs(outdir, exist_ok=True)
    baseline = np.bincount(y).max() / len(y)
    lines = [f"epochs: {len(y)}, subjects: {len(set(g))}",
             f"majority-class baseline accuracy: {baseline:.3f}",
             f"accuracy: {accuracy_score(y, pred):.3f}",
             f"macro F1: {f1_score(y, pred, average='macro'):.3f}",
             f"Cohen's kappa: {cohen_kappa_score(y, pred):.3f}", "",
             classification_report(y, pred, labels=range(5), target_names=NAMES, digits=3, zero_division=0)]
    text = "\n".join(lines)
    print(text)
    open(os.path.join(outdir, "metrics.txt"), "w").write(text)

    per = pd.DataFrame({"subject": g, "ok": y == pred}).groupby("subject")["ok"].mean().rename("accuracy")
    per.to_csv(os.path.join(outdir, "per_subject_accuracy.csv"))

    cm = confusion_matrix(y, pred, labels=range(5), normalize="true")
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(5), NAMES); ax.set_yticks(range(5), NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True (row-normalised)")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color="white" if cm[i, j] > .5 else "black")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "confusion_matrix.png"), dpi=200); plt.close(fig)

    s0 = sorted(set(g))[0]
    m = g == s0
    fig, ax = plt.subplots(2, 1, figsize=(9, 4), sharex=True)
    for a, data, title in zip(ax, (y[m], pred[m]), ("True hypnogram", "Predicted hypnogram")):
        a.step(np.arange(m.sum()) * 0.5 / 60, data, where="post"); a.set_title(f"{title} (subject {s0})", fontsize=9)
        a.set_yticks(range(5), NAMES); a.invert_yaxis()
    ax[1].set_xlabel("Hours"); fig.tight_layout()
    fig.savefig(os.path.join(outdir, "hypnogram_example.png"), dpi=200); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = build_dataset(a.subjects)
    data.to_csv(os.path.join(a.out, "features.csv"), index=False)
    y, pred, g = evaluate(data)
    report(y, pred, g, a.out)
