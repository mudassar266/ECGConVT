
from tensorflow.keras.layers import (
    Input, Concatenate, Dense, Dropout
)
from tensorflow.keras.models import Model

from src.xception_model import build_xception_feature_extractor
from src.vit_model      import build_vit_feature_extractor

def build_fusion_model(
    num_classes,
    xcep_weights="imagenet",
    vit_url="https://tfhub.dev/google/vit_base_patch16_224/feature_vector/1",
    trainable_backbones=False,
    mlp_units=(512, 256),
    dropout_rate=0.5
):
    """
    Creates the end‑to‑end fusion model:
      [299×299×3] → {Xception, ViT} → concat → MLP → softmax
    """
    inp = Input(shape=(299, 299, 3), name="image_input")
    # feature extractors
    xcep = build_xception_feature_extractor(
        input_shape=(299, 299, 3),
        weights=xcep_weights,
        trainable=trainable_backbones
    )
    vit = build_vit_feature_extractor(
        input_shape=(299, 299, 3),
        model_url=vit_url,
        trainable=trainable_backbones
    )

    xcep_feat = xcep(inp)
    vit_feat  = vit(inp)

    x = Concatenate(name="feature_concat")([xcep_feat, vit_feat])
    # MLP head
    for i, units in enumerate(mlp_units, 1):
        x = Dense(units, activation="relu", name=f"mlp_dense_{i}")(x)
        x = Dropout(dropout_rate,   name=f"mlp_dropout_{i}")(x)

    outputs = Dense(num_classes, activation="softmax", name="classifier")(x)
    return Model(inputs=inp, outputs=outputs, name="fusion_model")
