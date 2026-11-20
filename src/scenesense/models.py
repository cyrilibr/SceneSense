"""The two architectures used in SceneSense."""
import keras
from keras import layers

from scenesense.config import CLASSES


def augmentation_layers(inputs):
    """Transforms apply only during fit, never at inference."""
    x = layers.RandomFlip("horizontal", name="horizontal_flip")(inputs)
    x = layers.RandomRotation(0.05, name="small_rotation")(x)
    x = layers.RandomZoom(0.10, name="small_zoom")(x)
    return x


def build_custom_cnn(augment=False, dropout=False, image_size=160):
    inputs = keras.Input(shape=(image_size, image_size, 3), name="image")
    x = augmentation_layers(inputs) if augment else inputs
    x = layers.Rescaling(1.0 / 255.0, name="normalize_0_1")(x)
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"conv_{filters}")(x)
        x = layers.MaxPooling2D(pool_size=2, name=f"pool_{filters}")(x)
    x = layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = layers.Dense(128, activation="relu", name="dense_128")(x)
    if dropout:
        x = layers.Dropout(0.30, name="dropout_30")(x)
    outputs = layers.Dense(len(CLASSES), activation="softmax", name="class_probabilities")(x)
    return keras.Model(inputs, outputs, name="custom_cnn")


def build_mobilenet(image_size=160, pretrained=True):
    """ImageNet backbone, six-class head, and correct MobileNetV2 normalization."""
    inputs = keras.Input(shape=(image_size, image_size, 3), name="image")
    x = augmentation_layers(inputs)
    # MobileNetV2 expects values in [-1, 1], not [0, 1].
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenet_preprocess")(x)
    base = keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet" if pretrained else None,
    )
    base.trainable = False
    # BatchNorm remains in inference mode even during subsequent fine-tuning.
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = layers.Dropout(0.20, name="dropout_20")(x)
    outputs = layers.Dense(len(CLASSES), activation="softmax", name="class_probabilities")(x)
    return keras.Model(inputs, outputs, name="mobilenetv2_scene_classifier")


def enable_fine_tuning(model, last_layers=25):
    base = next((layer for layer in model.layers
                 if isinstance(layer, keras.Model) and "mobilenet" in layer.name.lower()), None)
    if base is None:
        raise ValueError("MobileNetV2 backbone not found")
    if last_layers < 1:
        raise ValueError("last_layers must be positive")
    base.trainable = True
    cutoff = max(0, len(base.layers) - last_layers)
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= cutoff and not isinstance(layer, layers.BatchNormalization)
    return sum(int(layer.trainable) for layer in base.layers)


def compile_model(model, learning_rate=1e-3):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def trainable_parameter_count(model):
    import numpy as np
    return int(sum(np.prod(weight.shape) for weight in model.trainable_weights))
