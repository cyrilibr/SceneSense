"""One-epoch smoke test. The tiny synthetic dataset cannot measure scientific accuracy."""
from scenesense.config import Settings
from scenesense.data import build_manifest
from scenesense.evaluation import evaluate_one, plot_training, select_model
from scenesense.training import checkpoint_path, train_experiment


def test_train_save_evaluate_cnn(small_dataset, tmp_path):
    cfg = Settings(data_dir=small_dataset, output_dir=tmp_path / "out", size=64,
                   batch_size=4, epochs_cnn=1, patience=1)
    build_manifest(cfg.data_dir, cfg.manifest_path)
    info = train_experiment("cnn_baseline", cfg)
    assert info["epochs_completed"] == 1
    assert checkpoint_path(cfg.output_dir, "cnn_baseline").is_file()
    assert select_model(cfg) == "cnn_baseline"
    assert plot_training(cfg).is_file()
    metrics = evaluate_one(cfg, "cnn_baseline", "test")
    assert metrics["samples"] == 12
    assert 0 <= metrics["macro_f1"] <= 1
    assert (cfg.output_dir / "evaluation/test_cnn_baseline/confusion_matrix.png").is_file()
