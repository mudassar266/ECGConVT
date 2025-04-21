import tensorflow as tf
import os

def get_ecg_datasets(
    data_dir,
    img_size=(299, 299),
    batch_size=32,
    val_split=0.2,
    test_split=0.1,
    seed=42
):
    """
    Loads ECG image data from `data_dir`, splits into train/val/test, and returns
    tf.data.Dataset objects plus the class names.

    Args:
        data_dir (str): Path to root data folder with one subfolder per class.
        img_size (tuple): (height, width) to resize images to.
        batch_size (int): Batch size.
        val_split (float): Fraction of data for validation.
        test_split (float): Fraction of data for testing.
        seed (int): Random seed for shuffling/splitting.

    Returns:
        train_ds (Dataset), val_ds (Dataset), test_ds (Dataset), class_names (list)
    """
    # sanity check splits
    if val_split + test_split >= 1.0:
        raise ValueError("val_split + test_split must be < 1.0")

    # First split off train vs (val+test)
    train_frac = 1.0 - (val_split + test_split)
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="categorical",
        validation_split=val_split + test_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size
    )
    val_test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="categorical",
        validation_split=val_split + test_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size
    )

    # Now split val_test_ds into val and test
    total_val_test = tf.data.experimental.cardinality(val_test_ds).numpy()
    val_count = int(total_val_test * (val_split / (val_split + test_split)))
    val_ds = val_test_ds.take(val_count)
    test_ds = val_test_ds.skip(val_count)

    class_names = train_ds.class_names

    # Normalize pixel values to [0,1]
    normalization = tf.keras.layers.Rescaling(1.0 / 255)

    def prep(ds, shuffle=False):
        ds = ds.map(lambda x, y: (normalization(x), y), 
                    num_parallel_calls=tf.data.AUTOTUNE)
        if shuffle:
            ds = ds.shuffle(1000, seed=seed)
        return ds.cache().prefetch(tf.data.AUTOTUNE)

    train_ds = prep(train_ds, shuffle=True)
    val_ds   = prep(val_ds)
    test_ds  = prep(test_ds)

    return train_ds, val_ds, test_ds, class_names


if __name__ == "__main__":
    # quick sanity check
    data_root = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    train, val, test, names = get_ecg_datasets(data_root)
    print("Classes:", names)
    print("Train batches:", tf.data.experimental.cardinality(train).numpy())
    print("Val batches:  ", tf.data.experimental.cardinality(val).numpy())
    print("Test batches: ", tf.data.experimental.cardinality(test).numpy())
