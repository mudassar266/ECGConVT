
import os
import tensorflow as tf

from src.config       import load_config
from src.data_loader  import get_ecg_datasets
from src.fusion_model import build_fusion_model

def main(config_path="config.yaml"):
    cfg = load_config(config_path)

    # — Data —
    train_ds, val_ds, test_ds, class_names = get_ecg_datasets(
        data_dir   = cfg["data_dir"],
        img_size   = tuple(cfg["img_size"]),
        batch_size = cfg["batch_size"],
        val_split  = cfg["val_split"],
        test_split = cfg["test_split"],
        seed       = cfg["seed"]
    )

    # — Model —
    model = build_fusion_model(
        num_classes         = len(class_names),
        xcep_weights        = cfg["xcep_weights"],
        vit_url             = cfg["vit_url"],
        trainable_backbones = cfg["trainable_backbones"],
        mlp_units           = tuple(cfg["mlp_units"]),
        dropout_rate        = cfg["dropout_rate"]
    )

    optimizer = tf.keras.optimizers.get(cfg["optimizer"])
    optimizer.learning_rate = cfg["learning_rate"]

    model.compile(
        optimizer = optimizer,
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy",
                     tf.keras.metrics.Precision(name="precision"),
                     tf.keras.metrics.Recall(name="recall")]
    )

    # — Callbacks —
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath   = os.path.join(cfg["checkpoint_dir"], "best_model.h5"),
        save_best_only = True,
        monitor    = "val_loss"
    )
    cbs = [ckpt_cb]
    if cfg.get("tensorboard_logdir"):
        tb_cb = tf.keras.callbacks.TensorBoard(log_dir=cfg["tensorboard_logdir"])
        cbs.append(tb_cb)

    # — Train —
    history = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = cfg["epochs"],
        callbacks       = cbs
    )

    # — Save history & evaluate —
    if cfg.get("history_path"):
        import json
        with open(cfg["history_path"], "w") as f:
            json.dump(history.history, f, indent=2)

    results = model.evaluate(test_ds, verbose=2)
    print({name: val for name, val in zip(model.metrics_names, results)})

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", default="config.yaml",
        help="Path to config.yml"
    )
    args = parser.parse_args()
    main(args.config)
