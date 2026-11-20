"""Command-line entry point. Run: python -m scenesense.cli --help"""
import argparse
from pathlib import Path

import keras
import numpy as np
from PIL import Image, ImageOps

from scenesense.config import CLASSES, EXPERIMENTS, Settings
from scenesense.data import build_manifest
from scenesense.evaluation import evaluate_final, evaluate_one, load_validation_results, plot_training, select_model
from scenesense.training import checkpoint_path, train_all, train_experiment


def run(argv=None):
    parser = argparse.ArgumentParser(description="SceneSense — six-class image classification")
    parser.add_argument("--data-dir", default="data/intel-image-classification", type=Path)
    parser.add_argument("--output-dir", default="results", type=Path)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--size", default=160, type=int)
    parser.add_argument("--epochs-cnn", default=25, type=int)
    parser.add_argument("--epochs-transfer", default=15, type=int)
    parser.add_argument("--epochs-finetune", default=10, type=int)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="Create fixed, class-stratified train/validation/test manifest")
    tr = sub.add_parser("train", help="Train a single experiment")
    tr.add_argument("experiment", choices=EXPERIMENTS)
    sub.add_parser("train-all", help="Train all five experiments in order")
    ev = sub.add_parser("evaluate", help="Evaluate validation or selected test model")
    ev.add_argument("--experiment", choices=EXPERIMENTS)
    ev.add_argument("--split", choices=("validation", "test"), default="test")
    sub.add_parser("plot", help="Plot validation-loss histories")
    inf = sub.add_parser("predict", help="Classify a single image")
    inf.add_argument("image", type=Path)
    inf.add_argument("--experiment", choices=EXPERIMENTS)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.size < 32:
        parser.error("batch size must be positive and size must be at least 32")
    if min(args.epochs_cnn, args.epochs_transfer, args.epochs_finetune) < 1:
        parser.error("epoch counts must all be positive")

    settings = Settings(data_dir=args.data_dir, output_dir=args.output_dir,
                        batch_size=args.batch_size, size=args.size,
                        epochs_cnn=args.epochs_cnn,
                        epochs_transfer=args.epochs_transfer,
                        epochs_finetune=args.epochs_finetune)
    if args.command == "prepare":
        result = build_manifest(settings.data_dir, settings.manifest_path,
                                seed=settings.seed, validation_fraction=settings.validation_fraction)
        print(result.groupby(["split", "label"]).size().unstack(fill_value=0).rename(columns=dict(enumerate(CLASSES))))
        print(f"Saved: {settings.manifest_path}")
    elif args.command == "train":
        print(train_experiment(args.experiment, settings))
    elif args.command == "train-all":
        print(train_all(settings))
    elif args.command == "evaluate":
        if args.experiment:
            print(evaluate_one(settings, args.experiment, args.split))
        elif args.split == "test":
            print(evaluate_final(settings))
        else:
            exp = select_model(settings)
            print(evaluate_one(settings, exp, "validation"))
    elif args.command == "plot":
        print(plot_training(settings))
        print(load_validation_results(settings)[["experiment", "validation_accuracy", "validation_loss"]])
    elif args.command == "predict":
        exp = args.experiment or select_model(settings)
        model = keras.models.load_model(checkpoint_path(settings.output_dir, exp))
        if not args.image.is_file():
            raise FileNotFoundError(args.image)
        with Image.open(args.image) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            rgb = rgb.resize((settings.size, settings.size), Image.Resampling.BILINEAR)
            batch = np.asarray(rgb, dtype=np.float32)[None, ...]
        proba = np.asarray(model.predict(batch, verbose=0))[0]
        ranked = np.argsort(proba)[::-1]
        print(f"Model: {exp}")
        for index in ranked:
            print(f"{CLASSES[index]:12s} {proba[index]:.1%}")


if __name__ == "__main__":
    run()
