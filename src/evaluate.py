
import os
import numpy as np
import itertools
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

def plot_confusion_matrix(cm, classes, normalize=False, save_path=None):
    plt.figure(figsize=(6,6))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()

def evaluate_model(model_path, test_ds, class_names, output_dir="reports"):
    # Load trained model
    model = tf.keras.models.load_model(model_path)

    # Gather predictions & labels
    y_true, y_pred = [], []
    for x_batch, y_batch in test_ds:
        preds = model.predict(x_batch)
        y_true.extend(np.argmax(y_batch.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Classification report
    print(classification_report(y_true, y_pred, target_names=class_names))
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )
    os.makedirs(output_dir, exist_ok=True)
    import json
    with open(os.path.join(output_dir, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Confusion matrices
    cm_raw = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(
        cm_raw, class_names,
        normalize=False,
        save_path=os.path.join(output_dir, "cm_raw.png")
    )
    plot_confusion_matrix(
        cm_raw, class_names,
        normalize=True,
        save_path=os.path.join(output_dir, "cm_norm.png")
    )
