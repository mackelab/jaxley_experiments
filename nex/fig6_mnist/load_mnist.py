import tensorflow_datasets as tfds
import pickle

data_dir = '.'

# Fetch full datasets for evaluation
# tfds.load returns tf.Tensors (or tf.data.Datasets if batch_size != -1)
# You can convert them to NumPy arrays (or iterables of NumPy arrays) with tfds.dataset_as_numpy
mnist_data, info = tfds.load(name="mnist", batch_size=-1, data_dir=data_dir, with_info=True)
mnist_data = tfds.as_numpy(mnist_data)

with open("results/mnist_data.pickle", "wb") as handle:
    pickle.dump(mnist_data, handle)