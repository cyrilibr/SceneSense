"""Controlled comparisons with one saved checkpoint per experiment."""
import json
import time
from pathlib import Path

import keras
import pandas as pd

from scenesense.config import EXPERIMENTS, Settings
from scenesense.data import make_batches, read_manifest
from scenesense.models import (
    build_custom_cnn, build_mobilenet, compile_model,
    enable_fine_tuning, trainable_parameter_count,
)


def checkpoint_path(output_dir, experiment):
    return Path(output_dir) / "models" / f"{experiment}.keras"


def _callbacks(path, patience):
    return [
        keras.callbacks.ModelCheckpoint(str(path), monitor="val_loss", mode="min",
                                        save_best_only=True, verbose=0),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=2, min_lr=1e-6),
        keras.callbacks.TerminateOnNaN(),
    ]


def _write_info(folder, experiment, info, history):
    pd.DataFrame(history.history).to_csv(folder / f"{experiment}_history.csv", index=False)
    with (folder / f"{experiment}_info.json").open("w", encoding="utf-8") as file:
        json.dump(info, file, indent=2)


def train_experiment(name: str, settings: Settings, pretrained=True):
    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {name}. Choose from {', '.join(EXPERIMENTS)}")
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    frame = read_manifest(settings.data_dir, settings.manifest_path)
    train_data = make_batches(frame, settings.data_dir, "train", settings)
    val_data = make_batches(frame, settings.data_dir, "validation", settings)
    keras.utils.set_random_seed(settings.seed)

    if name.startswith("cnn_"):
        model = build_custom_cnn(
            augment=name != "cnn_baseline",
            dropout=name == "cnn_dropout",
            image_size=settings.size,
        )
        epochs, lr = settings.epochs_cnn, settings.learning_rate
    elif name == "mobilenet_frozen":
        model = build_mobilenet(settings.size, pretrained=pretrained)
        epochs, lr = settings.epochs_transfer, settings.learning_rate
    else:
        parent = checkpoint_path(settings.output_dir, "mobilenet_frozen")
        if not parent.is_file():
            raise FileNotFoundError("Train mobilenet_frozen before mobilenet_finetune")
        model = keras.models.load_model(parent)
        enabled = enable_fine_tuning(model, last_layers=25)
        if enabled == 0:
            raise RuntimeError("No MobileNetV2 layers were unfrozen for fine-tuning")
        epochs, lr = settings.epochs_finetune, settings.fine_tune_learning_rate

    compile_model(model, learning_rate=lr)
    destination = checkpoint_path(settings.output_dir, name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=_callbacks(destination, settings.patience),
        verbose=2,
    )
    duration = round(time.perf_counter() - start, 2)
    best = keras.models.load_model(destination)
    validation_loss, validation_accuracy = best.evaluate(val_data, verbose=0)
    info = {
        "experiment": name,
        "trained": True,
        "pretrained_weights": "imagenet" if name.startswith("mobilenet") and pretrained else "none",
        "train_images": int((frame.split == "train").sum()),
        "validation_images": int((frame.split == "validation").sum()),
        "test_images_reserved": int((frame.split == "test").sum()),
        "epochs_completed": len(history.history["loss"]),
        "duration_seconds": duration,
        "trainable_parameters": trainable_parameter_count(best),
        "validation_loss": round(float(validation_loss), 5),
        "validation_accuracy": round(float(validation_accuracy), 5),
        "image_size": settings.size,
        "batch_size": settings.batch_size,
        "seed": settings.seed,
    }
    _write_info(settings.output_dir, name, info, history)
    print(f"\nSaved {name} → {destination}")
    return info


def train_all(settings: Settings):
    summaries = []
    for name in EXPERIMENTS:
        summaries.append(train_experiment(name, settings))
    pd.DataFrame(summaries).to_csv(settings.output_dir / "validation_summary.csv", index=False)
    return summaries
