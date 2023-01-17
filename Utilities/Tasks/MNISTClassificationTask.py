# fmt: off
from typing import List

from .TensorFlowDatasetTask import TensorFlowDatasetTask

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import tensorflow_datasets as tfds
# fmt: on

class MNISTClassificationTask(TensorFlowDatasetTask):
    """
    Create a new task based around classifying between different MNIST digit images
    The MNIST data is taken from tensorflow_datasets and is processed slightly to 
    improve performance

    Note for this task, the dataset originally has numeric labels (not one-hot)
    In the create dataset method we map these labels to one_hot to make modelling easier
    For example, a task that is classifying between the digits 3 and 4 does not need 10 outputs 
    It is recommended to use CategoricalLoss or BinaryCategoricalLoss
    """

    IMAGE_SIZE = (28,28,1)

    def __init__(self, 
            name: str,
            model: tf.keras.models.Model,
            model_base_loss: tf.keras.losses.Loss,
            task_labels: List[int],
            training_batches: int = 0,
            validation_batches: int = 0,
            batch_size: int = 32,
            **kwargs,
        ) -> None:
        """
        Create a new MNIST classification task.

        Parameters:
            name: str
                The name of this task. Usually like "Task 1"

            model: tf.keras.models.Model
                The model to fit to the tasks data

            model_base_loss: tf.keras.losses.Loss:
                The base loss function of the model (before EWC)

            task_labels: List[int]
                The digits to differentiate in this task
                Usually a list of two digits (e.g. [0,1]) for binary classification
                But can be larger (e.g. [0,1,2,3]) for a larger classification task
                Must be valid MNIST digits (0-9), list passed to dataset.filter

            training_batches: int
                The number of batches in the training dataset
                If 0 (default) use all batches available
            
            validation_batches: int
                The number of batches in the validation dataset
                If 0 (default) use all batches available

            batch_size: int
                The batch size for datasets
                Defaults to 128.

            **kwargs
                Other keyword arguments to be passed to super()
                Anything in this set is optional for this task 
                e.g. optional SequentialTask parameters
        """

        super().__init__(
            name = name,
            dataset_name="MNIST",
            model = model,
            model_base_loss = model_base_loss,
            task_labels=task_labels,
            training_batches = training_batches,
            validation_batches = validation_batches,
            batch_size=batch_size,
            **kwargs)
