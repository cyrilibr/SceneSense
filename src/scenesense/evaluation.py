"""Validation comparisons and a separate, one-time final test evaluation."""
import json
from pathlib import Path

import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from scenesense.config import CLASSES, EXPERIMENTS, Settings
from scenesense.data import make_batches, read_manifest
from scenesense.training import checkpoint_path


def load_validation_results(settings):
    results = []
    for exp in EXPERIMENTS:
        p = settings.output_dir / f"{exp}_info.json"
        if not p.is_file():
            continue
        with p.open(encoding="utf-8") as f:
            results.append(json.load(f))
    if not results:
        raise FileNotFoundError("No trained models found. Run train-all first.")
    return pd.DataFrame(results)


def select_model(settings):
    """Select by validation loss without consulting the test split."""
    frame = load_validation_results(settings)
    return str(frame.sort_values(["validation_loss", "experiment"]).iloc[0]["experiment"])


def plot_training(settings):
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    found = False
    for exp in EXPERIMENTS:
        path = settings.output_dir / f"{exp}_history.csv"
        if not path.is_file():
            continue
        hist = pd.read_csv(path)
        ax.plot(range(1, len(hist) + 1), hist.val_loss, marker="o", markersize=2,
                linewidth=1.6, label=exp.replace("_", " "))
        found = True
    if not found:
        plt.close(fig)
        raise FileNotFoundError("No training histories found.")
    ax.set(xlabel="Epoch", ylabel="Validation loss", title="Validation loss by experiment")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = settings.output_dir / "validation_curves.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def evaluate_one(settings: Settings, experiment: str, split="test"):
    """Never called during model fitting; keeps the final test set separate."""
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    model_path = checkpoint_path(settings.output_dir, experiment)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint missing: {model_path}")
    frame = read_manifest(settings.data_dir, settings.manifest_path)
    subset = frame.loc[frame.split == split].reset_index(drop=True)
    batches = make_batches(frame, settings.data_dir, split, settings)
    model = keras.models.load_model(model_path)
    proba = model.predict(batches, verbose=0)
    pred = np.asarray(proba).argmax(axis=1)
    expected = subset.label.to_numpy(dtype=int)
    if len(pred) != len(expected):
        raise RuntimeError("Predictions and labels have different lengths")
    metrics = {
        "experiment": experiment, "split": split, "samples": len(expected),
        "accuracy": round(float(accuracy_score(expected, pred)), 5),
        "macro_f1": round(float(f1_score(expected, pred, average="macro", zero_division=0)), 5),
    }
    out = settings.output_dir / "evaluation" / f"{split}_{experiment}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report = classification_report(expected, pred, labels=list(range(len(CLASSES))),
                                   target_names=list(CLASSES), digits=4, zero_division=0)
    (out / "classification_report.txt").write_text(report, encoding="utf-8")
    cm = confusion_matrix(expected, pred, labels=list(range(len(CLASSES))))
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    display = ax.imshow(cm, cmap="Blues")
    fig.colorbar(display, ax=ax, shrink=0.78)
    ax.set(xticks=range(6), yticks=range(6), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Predicted", ylabel="Actual", title=f"Confusion matrix — {experiment}")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    for row in range(6):
        for col in range(6):
            ax.text(col, row, str(cm[row, col]), ha="center", va="center",
                    color="white" if cm[row, col] > cm.max() / 2 else "black", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    examples = subset.copy()
    examples["prediction"] = pred
    examples["correct"] = pred == expected
    examples["confidence"] = np.asarray(proba).max(axis=1)
    examples.to_csv(out / "predictions.csv", index=False)
    return metrics


def evaluate_final(settings: Settings):
    exp = select_model(settings)
    result = evaluate_one(settings, exp, split="test")
    print(f"Selected on validation loss: {exp}")
    print(f"Final test accuracy: {result['accuracy']:.4f}; macro F1: {result['macro_f1']:.4f}")
    return result
