
import os
import matplotlib.pyplot as plt

def plot_history(history, output_dir="reports"):
    """
    Plot and save training & validation loss/accuracy curves.
    Expects history.history dict with keys:
      - loss, val_loss
      - accuracy, val_accuracy
    """
    os.makedirs(output_dir, exist_ok=True)

    # Loss
    plt.figure()
    plt.plot(history["loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

    # Accuracy
    plt.figure()
    plt.plot(history["accuracy"], label="train_acc")
    plt.plot(history["val_accuracy"], label="val_acc")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "accuracy_curve.png"))
    plt.close()
