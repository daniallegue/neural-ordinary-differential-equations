import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np


def get_mnist_dataloader(batch_size: int = 128, split: str = 'train', shuffle: bool = True) -> np.ndarray:
    """
    Returns an iterator for the MNIST dataset.

    :param batch_size (int): Number of samples per batch.
    :param split (str): Dataset split to load ('train' or 'test').
    :param shuffle (bool): Whether to shuffle the dataset.

    :returns Iterator yielding batches of (images, labels)
    """

    ds = tfds.load('mnist', split=split, as_supervised=True)
    if shuffle:
        ds = ds.shuffle(10000)
    # Normalize images to [0,1]
    ds = ds.map(lambda img, label: (tf.cast(img, tf.float32) / 255.0, label))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return tfds.as_numpy(ds)


def get_cifar10_dataloader(batch_size: int = 128, split: str = 'train', shuffle: bool = True) -> np.ndarray:
    """
    Returns an iterator for the CIFAR-10 dataset.

    :param batch_size (int): Number of samples per batch.
    :param split (str): Dataset split to load ('train' or 'test').
    :param shuffle (bool): Whether to shuffle the dataset.

    :returns Iterator yielding batches of (images, labels)
    """
    ds = tfds.load('cifar10', split=split, as_supervised=True)
    if shuffle:
        ds = ds.shuffle(10000)
    ds = ds.map(lambda img, label: (tf.cast(img, tf.float32) / 255.0, label))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return tfds.as_numpy(ds)