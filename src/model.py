"""Model definitions. Both models take raw pixels (0-255) so the API needs no extra preprocessing."""
import tensorflow as tf

layers = tf.keras.layers
models = tf.keras.models


def build_cnn(num_classes, img_size=224, dropout=0.3):
    """Simple CNN baseline: 3 conv blocks + dense head."""
    return models.Sequential(
        [
            layers.Input((img_size, img_size, 3)),
            layers.Rescaling(1 / 255.0),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="simple_cnn",
    )


def build_mobilenet(num_classes, img_size=224, dropout=0.2, pretrained=True):
    """MobileNetV2 transfer learning: frozen ImageNet base + new classification head."""
    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet" if pretrained else None,
    )
    base.trainable = False   # freeze: only the new head is trained

    inputs = layers.Input((img_size, img_size, 3))
    x = layers.Rescaling(1 / 127.5, offset=-1)(inputs)   # MobileNetV2 expects [-1, 1]
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name="mobilenetv2_transfer")


def build_model(name, num_classes, img_size, cfg):
    if name == "cnn":
        return build_cnn(num_classes, img_size, cfg["dropout"])
    if name == "mobilenet":
        return build_mobilenet(num_classes, img_size, cfg["dropout"], cfg.get("pretrained", True))
    raise ValueError(f"Unknown model: {name}")