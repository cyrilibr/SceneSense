import numpy as np
import keras

from scenesense.models import (
    build_custom_cnn, build_mobilenet, compile_model, enable_fine_tuning,
    trainable_parameter_count,
)


def test_three_cnn_variants():
    x = np.random.default_rng(10).integers(0, 255, (2, 64, 64, 3)).astype("float32")
    for aug, drop in ((False, False), (True, False), (True, True)):
        keras.utils.set_random_seed(42)
        model = compile_model(build_custom_cnn(aug, drop, image_size=64))
        y = model(x, training=False)
        assert tuple(y.shape) == (2, 6)
        np.testing.assert_allclose(keras.ops.convert_to_numpy(y).sum(axis=1), [1, 1], atol=1e-5)
        assert trainable_parameter_count(model) > 0


def test_mobilenet_frozen_and_fine_tuned_without_downloading_weights():
    model = build_mobilenet(image_size=64, pretrained=False)
    base = next(x for x in model.layers if isinstance(x, keras.Model) and "mobilenet" in x.name)
    assert not base.trainable
    assert sum(layer.trainable for layer in base.layers) == 0
    x = np.zeros((1, 64, 64, 3), dtype=np.float32)
    assert tuple(model(x, training=False).shape) == (1, 6)
    count = enable_fine_tuning(model, 25)
    assert count > 0
    assert all(not layer.trainable for layer in base.layers
               if isinstance(layer, keras.layers.BatchNormalization))
    compile_model(model, learning_rate=1e-5)
    assert trainable_parameter_count(model) > 0
