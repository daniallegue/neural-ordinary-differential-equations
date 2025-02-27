import tensorflow as tf
import numpy as np


def get_mnist_dataloader(batch_size: int = 128, split: str = 'train', shuffle: bool = True, limit : int = None):
    """
    Returns an iterator for the MNIST dataset using tf.keras.datasets.

    :param batch_size: Number of samples per batch.
    :param split: Which split to load ('train' or 'test').
    :param shuffle: Whether to shuffle the dataset.
    :returns: Iterator yielding batches of (images, labels) as NumPy arrays.
    """
    if split == 'train':
        (images, labels), _ = tf.keras.datasets.mnist.load_data()
    elif split == 'test':
        _, (images, labels) = tf.keras.datasets.mnist.load_data()
    else:
        raise ValueError("split must be 'train' or 'test'")

    images = images.astype(np.float32) / 255.0


    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(images))
    if limit is not None:
        dataset = dataset.take(limit)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset.as_numpy_iterator()


def get_cifar10_dataloader(batch_size: int = 128, split: str = 'train', shuffle: bool = True, limit : int = None):
    """
    Returns an iterator for the CIFAR-10 dataset using tf.keras.datasets.

    :param batch_size: Number of samples per batch.
    :param split: Which split to load ('train' or 'test').
    :param shuffle: Whether to shuffle the dataset.
    :returns: Iterator yielding batches of (images, labels) as NumPy arrays.
    """
    if split == 'train':
        (images, labels), _ = tf.keras.datasets.cifar10.load_data()
    elif split == 'test':
        _, (images, labels) = tf.keras.datasets.cifar10.load_data()
    else:
        raise ValueError("split must be 'train' or 'test'")

    # Normalize images to [0,1] and convert to float32
    images = images.astype(np.float32) / 255.0

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(images))
    if limit is not None:
        dataset = dataset.take(list)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset.as_numpy_iterator()
