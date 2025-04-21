
import tensorflow as tf
import tensorflow_hub as hub

def build_vit_feature_extractor(
    input_shape=(299, 299, 3),
    model_url="https://tfhub.dev/google/vit_base_patch16_224/feature_vector/1",
    trainable=False
):
    """
    Builds a Vision Transformer feature extractor from TF‑Hub.

    - Resizes inputs to 224×224
    - Normalizes pixel values
    - Feeds through the ViT backbone, returning a 1D feature vector
    """
    inputs = tf.keras.Input(shape=input_shape, name="vit_input")
    # resize & normalize to [0,1]
    x = tf.keras.layers.Resizing(224, 224, name="vit_resize")(inputs)
    x = tf.keras.layers.Rescaling(1.0/255, name="vit_rescale")(x)

    vit_layer = hub.KerasLayer(model_url, trainable=trainable, name="vit_hub")
    features = vit_layer(x)

    return tf.keras.Model(inputs=inputs, outputs=features, name="vit_feature_extractor")
