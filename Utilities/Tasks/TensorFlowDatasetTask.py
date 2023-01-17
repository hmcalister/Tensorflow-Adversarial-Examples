# fmt: off
import math
from typing import List, Tuple
from Utilities.Utils import normalize_img

from .GenericTask import GenericTask

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import tensorflow_datasets as tfds
# fmt: on

class TensorFlowDatasetTask(GenericTask):
    """
    An abstract class for tasks created from Tensorflow Datasets
    The data is taken from tensorflow_datasets and is processed slightly to 
    improve performance

    Note in TFDS, datasets originally have numeric labels (not one-hot)
    In the create dataset method we map these labels to one_hot to make modelling easier
    For example, a task that is classifying between the digits 3 and 4 does not need 10 outputs 
    It is recommended to use CategoricalLoss or BinaryCategoricalLoss
    """

    def __init__(self, 
            name: str,
            dataset_name: str,
            model: tf.keras.models.Model,
            model_base_loss: tf.keras.losses.Loss,
            task_labels: List[int],
            train_split_name: str = "train",
            training_batches: int = 0,
            validation_split_name: str = "test",
            validation_batches: int = 0,
            batch_size: int = 32,
            training_image_augmentation: tf.keras.Sequential | None = None,
            **kwargs,
        ) -> None:
        """
        Parameters:
            name: str
                The name of this task. Usually like "Task 1"

            dataset_name: str
                The name of the dataset to load

            model: tf.keras.models.Model
                The model to fit to the tasks data

            model_base_loss: tf.keras.losses.Loss:
                The base loss function of the model (before EWC)

            task_labels: List[int]
                The classes to differentiate in this task
                Usually a list of two digits (e.g. [0,1]) for binary classification
                But can be larger (e.g. [0,1,2,3]) for a larger classification task

            training_batches: int
                The number of batches in the training dataset
                If 0 (default) use all batches available
            
            validation_batches: int
                The number of batches in the validation dataset
                If 0 (default) use all batches available

            batch_size: int
                The batch size for datasets
                Defaults to 128.
            
            training_image_augmentation: tf.keras.Sequential | None
                A pipeline to augment the training images before training
                e.g. add random resizing, zooming, ...
                See https://pyimagesearch.com/2021/06/28/data-augmentation-with-tf-data-and-tensorflow/
                If None no augmentation is applied

            **kwargs
                Other keyword arguments to be passed to super()
                Anything in this set is optional for this task 
                e.g. optional SequentialTask parameters
        """

        self.task_labels = task_labels
        self.batch_size = batch_size
        self.training_batches = training_batches 
        self.validation_batches = validation_batches
        self.training_image_augmentation: tf.keras.Sequential = training_image_augmentation

        (self.full_training_dataset, self.full_validation_dataset), ds_info = tfds.load(
            dataset_name,
            split=[train_split_name, validation_split_name],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
        )
        self.full_training_dataset: tf.data.Dataset = self.full_training_dataset.map(normalize_img,num_parallel_calls=tf.data.AUTOTUNE)
        self.full_validation_dataset: tf.data.Dataset = self.full_validation_dataset.map(normalize_img,num_parallel_calls=tf.data.AUTOTUNE)
        (training_dataset, validation_dataset) = self.create_datasets()
        del(self.full_training_dataset)
        del(self.full_validation_dataset)
        
        super().__init__(
            name = name,
            model = model,
            model_base_loss = model_base_loss,
            training_dataset = training_dataset,
            training_batches = self.training_batches,
            validation_dataset = validation_dataset,
            validation_batches = self.validation_batches,
            **kwargs)

    def create_datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Creates (and returns) a tuple of (training_dataset, validation_dataset)
        based on the Fashion MNIST dataset
        """

        filter_range = tf.constant(self.task_labels, dtype=tf.int64)
        one_hot_depth = len(self.task_labels)

        # We need to filter out only the labels we actually want before preprocessing
        training_dataset = self.full_training_dataset \
            .filter(lambda _, label: tf.reduce_any(tf.equal(label, filter_range)))

        # This is an ugly hack to map numbers to cardinal values
        # e.g. if task digits are (7,4,5) this loop maps the results to (0,1,2)
        # ideally combine this loop and mapping to the map below (one-hot encoding) but... it works
        for final_val, init_val in enumerate(self.task_labels):
            final_tensor = tf.constant(final_val, dtype=tf.int64)
            training_dataset = training_dataset.map(lambda x,y: (x, final_tensor if y==init_val else y))

        if self.training_batches == 0:
            training_samples = sum(1 for _ in training_dataset)
            self.training_batches = int(training_samples/self.batch_size)
        else:
            training_samples = self.training_batches * self.batch_size

        if self.training_image_augmentation is not None:
            training_dataset = training_dataset.map(lambda x,y: (self.training_image_augmentation(x), y))

        training_dataset = training_dataset \
            .map(lambda x,y: (x,tf.one_hot(y, depth=one_hot_depth))) \
            .take(training_samples) \
            .shuffle(training_samples) \
            .batch(self.batch_size) \
            .repeat() \
            .prefetch(tf.data.experimental.AUTOTUNE)

        # Repeat the same with validation
        validation_dataset = self.full_validation_dataset \
            .filter(lambda _, label: tf.reduce_any(tf.equal(label, filter_range)))

        for final_val, init_val in enumerate(self.task_labels):
            final_tensor = tf.constant(final_val, dtype=tf.int64)
            validation_dataset = validation_dataset.map(lambda x,y: (x, final_tensor if y==init_val else y))

        if self.validation_batches == 0:
            validation_samples = sum(1 for _ in validation_dataset)
            self.validation_batches = int(validation_samples/self.batch_size)
        else:
            validation_samples = self.validation_batches * self.batch_size
            
        validation_dataset = validation_dataset \
            .map(lambda x,y: (x,tf.one_hot(y, depth=one_hot_depth))) \
            .take(validation_samples) \
            .shuffle(validation_samples) \
            .batch(self.batch_size) \
            .repeat() \
            .prefetch(tf.data.experimental.AUTOTUNE)
        
        return (training_dataset, validation_dataset)