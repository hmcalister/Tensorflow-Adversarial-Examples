# fmt: off
from typing import List, Tuple
from Utilities.Utils import normalize_img
import numpy as np
import pandas as pd

from .GenericTask import GenericTask

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
# fmt: on

class IntelNaturalScenesClassificationTask(GenericTask):
    """
    Create a new IntelNaturalScenes Classification Task
    This dataset is based around classifying scenes based on images
    Interestingly these images are large and non-uniform

    https://www.kaggle.com/datasets/puneet6060/intel-image-classification
    """

    IMAGE_SIZE = (150,150,3)

    def __init__(self, 
            name: str,
            model: tf.keras.models.Model,
            model_base_loss: tf.keras.losses.Loss,
            task_labels: List[int],
            batch_size: int = 32,
            training_batches: int = 0,
            validation_batches: int = 0,
            training_image_augmentation: tf.keras.Sequential | None = None,
            data_path: str = "datasets/IntelNaturalScenes",
            image_size = (150,150),
            **kwargs,
        ) -> None:
        """
        Create a new IntelNaturalScenes classification task.

        Parameters:
            name: str
                The name of this task. Usually like "Task 1"

            model: tf.keras.models.Model
                The model to fit to the tasks data

            model_base_loss: tf.keras.losses.Loss:
                The base loss function of the model (before EWC)
            
            task_labels: List[int]
                The classes to differentiate in this task
                Usually a list of two digits (e.g. [0,1]) for binary classification
                But can be larger (e.g. [0,1,2,3]) for a larger classification task
                Can be from the list:
                    0 "buildings"
                    1 "forest"
                    2 "glacier"
                    3 "mountain"
                    4 "sea"
                    5 "street"
            
            batch_size: int
                The batch size for training

            training_batches: int
                The number of batches to take for training
                If 0 (default) use as many batches as possible

            validation_batches: int
                The number of batches to take for validation
                If 0 (default) use as many batches as possible

            training_image_augmentation: tf.keras.Sequential | None
                A pipeline to augment the training images before training
                e.g. add random resizing, zooming, ...
                See https://pyimagesearch.com/2021/06/28/data-augmentation-with-tf-data-and-tensorflow/
                If None no augmentation is applied

            data_path: str
                The path to the IntelNaturalScenes Dataset
                Should contain three unzipped folders
                seg_train, seg_test, seg_pred

            image_size: Tuple(int, int)
                The image size to pass to dataset loader
                Sets how big images will be once loaded
            """

        self.task_labels = task_labels
        self.batch_size = batch_size
        self.training_batches = training_batches 
        self.validation_batches = validation_batches
        self.training_image_augmentation: tf.keras.Sequential = training_image_augmentation
        self.data_path = data_path
        self.IMAGE_SIZE = image_size

        training_dataset, validation_dataset = self._load_data()

        super().__init__(
            name = name,
            model = model,
            model_base_loss = model_base_loss,
            training_dataset = training_dataset,
            training_batches = self.training_batches,
            validation_dataset = validation_dataset,
            validation_batches = self.validation_batches,
            **kwargs)


    def _load_data(self):
        """
        Loads the data from the specified data path
        If any augmentation specified apply this too
        """

        training_data_path = os.path.join(self.data_path, "seg_train", "seg_train")
        training_dataset: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
            training_data_path,
            validation_split=0,
            batch_size=None, #type: ignore
            label_mode="int",
            image_size=self.IMAGE_SIZE
        )

        filter_range = tf.constant(self.task_labels, dtype=tf.int32)
        one_hot_depth = len(self.task_labels)

        # We need to filter out only the labels we actually want before preprocessing
        training_dataset = training_dataset \
            .filter(lambda _, label: tf.reduce_any(tf.equal(label, filter_range)))

        # This is an ugly hack to map numbers to cardinal values
        # e.g. if task digits are (7,4,5) this loop maps the results to (0,1,2)
        # ideally combine this loop and mapping to the map below (one-hot encoding) but... it works
        for final_val, init_val in enumerate(self.task_labels):
            final_tensor = tf.constant(final_val, dtype=tf.int32)
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
            .map(normalize_img,num_parallel_calls=tf.data.AUTOTUNE) \
            .take(training_samples) \
            .shuffle(training_samples) \
            .batch(self.batch_size) \
            .repeat() \
            .prefetch(tf.data.experimental.AUTOTUNE)



        validation_data_path = os.path.join(self.data_path, "seg_test", "seg_test")
        validation_dataset: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
            validation_data_path,
            validation_split=0,
            batch_size=None, #type: ignore
            label_mode="int",
            image_size=self.IMAGE_SIZE
        )
        # Repeat the same with validation
        validation_dataset = validation_dataset \
            .filter(lambda _, label: tf.reduce_any(tf.equal(label, filter_range)))

        for final_val, init_val in enumerate(self.task_labels):
            final_tensor = tf.constant(final_val, dtype=tf.int32)
            validation_dataset = validation_dataset.map(lambda x,y: (x, final_tensor if y==init_val else y))

        if self.validation_batches == 0:
            validation_samples = sum(1 for _ in validation_dataset)
            self.validation_batches = int(validation_samples/self.batch_size)
        else:
            validation_samples = self.validation_batches * self.batch_size
            
        validation_dataset = validation_dataset \
            .map(lambda x,y: (x,tf.one_hot(y, depth=one_hot_depth))) \
            .map(normalize_img,num_parallel_calls=tf.data.AUTOTUNE) \
            .take(validation_samples) \
            .shuffle(validation_samples) \
            .batch(self.batch_size) \
            .repeat() \
            .prefetch(tf.data.experimental.AUTOTUNE)
        
        return (training_dataset, validation_dataset)
