
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model

def build_xception_feature_extractor(
    input_shape=(299, 299, 3),
    weights="imagenet",
    trainable=False
):
    """
    Builds an Xception-based feature extractor.

    Args:
        input_shape (tuple): Size of input images, e.g. (299, 299, 3).
        weights (str or None): Pretrained weights to load; use "imagenet" or None.
        trainable (bool): Whether to fine-tune the Xception backbone.

    Returns:
        model (tf.keras.Model): Model that maps input images to a feature vector.
    """
    # Define the input tensor
    inputs = Input(shape=input_shape, name="xception_input")

    # Load Xception without classification head
    base = Xception(
        include_top=False,
        weights=weights,
        input_tensor=inputs
    )
    base.trainable = trainable

    # Global average pooling to get a 1D feature vector
    x = GlobalAveragePooling2D(name="xception_gap")(base.output)

    # Build the model
    model = Model(inputs=inputs, outputs=x, name="xception_feature_extractor")
    return model


if __name__ == "__main__":
    # Quick test: instantiate and print summary
    model = build_xception_feature_extractor()
    model.summary()
