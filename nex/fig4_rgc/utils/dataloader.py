import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset

import logging

log = logging.getLogger("rgc")

def build_dataloaders(
    noise_full, currents, labels, loss_weights, val_frac, num_test, batch_size
):
    """Return dataloaders for retinal ganglion cell task."""
    num_datapoints = len(noise_full)
    assert num_test < num_datapoints

    # Validation fraction is computed without considering the test set.
    test_frac = num_test / num_datapoints
    val_frac = val_frac * (1 - test_frac)
    num_train = int(num_datapoints * (1 - val_frac - test_frac))

    # Perform data splits.
    all_inds = np.random.permutation(np.arange(num_datapoints))
    test_inds = all_inds[:num_test]
    train_inds = all_inds[num_test:num_test+num_train]
    val_inds = all_inds[num_test+num_train:]

    log.info(f"Num train {len(train_inds)}, num val {len(val_inds)}, num test {len(test_inds)}")
    
    tf.random.set_seed(1)
    dataloader = Dataset.from_tensor_slices(
        (
            noise_full[train_inds],
            currents[train_inds],
            labels[train_inds],
            loss_weights[train_inds],
        )
    )
    dataloader = dataloader.shuffle(dataloader.cardinality()).batch(batch_size)

    # The dataloader for evaluating the correlation under the training set; also used
    # for computing the receptive fields.
    eval_dataloaders = {}
    for split, inds in zip(["val", "test", "train"], [val_inds, test_inds, train_inds]):
        eval_dataloaders[split] = Dataset.from_tensor_slices(
            (
                noise_full[inds],
                currents[inds],
                labels[inds],
                loss_weights[inds],
            )
        )
        eval_batch_size = np.minimum(128 * 8, len(inds))
        eval_dataloaders[split] = eval_dataloaders[split].batch(eval_batch_size)

    with open(f"{os.getcwd()}/data/train_inds.pkl", "wb") as handle:
        pickle.dump(train_inds, handle)
    with open(f"{os.getcwd()}/data/test_inds.pkl", "wb") as handle:
        pickle.dump(test_inds, handle)
    with open(f"{os.getcwd()}/data/val_inds.pkl", "wb") as handle:
        pickle.dump(val_inds, handle)
    return dataloader, eval_dataloaders