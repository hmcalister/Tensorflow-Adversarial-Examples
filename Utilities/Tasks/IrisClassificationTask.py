# fmt: off
from typing import List, Tuple
import numpy as np
import pandas as pd

from .GenericTask import GenericTask

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
# fmt: on

class IrisClassificationTask(GenericTask):
    """
    Create a new classification task based on the Iris dataset
    Task consists of some subset of feature columns
    (['sepallength', 'sepalwidth', 'petallength', 'petalwidth'])
    being mapped to one-hot encoded label columns
    Loss should be categorical cross-entropy or similar for classification

    Warning! This dataset has only 150 items!
    Suggested training items is 120 and validation 30
    (Note: default batch_size=10, so choose training_batches=12, validation=3)
    """
    def __init__(self, 
            name: str,
            model: tf.keras.models.Model,
            model_base_loss: tf.keras.losses.Loss,
            feature_column_names: List[str],
            training_batches: int = 12,
            validation_batches: int = 3,
            batch_size: int = 10,
            iris_dataset_csv_path: str = "datasets/iris_csv.csv",
            **kwargs,
        ) -> None:
        """
        Create a new IRIS classification task.

        Parameters:
            name: str
                The name of this task. Usually like "Task 1"

            model: tf.keras.models.Model
                The model to fit to the tasks data

            model_base_loss: tf.keras.losses.Loss:
                The base loss function of the model (before EWC)

            feature_column_names: List[str]
                The column names to be used as features in this task
                Must be valid column names for the iris dataset
                e.g. 
                    'sepallength', 
                    'sepalwidth', 
                    'petallength', 
                    'petalwidth'

            training_batches: int
                The number of batches in the training dataset
            
            validation_batches: int
                The number of batches in the validation dataset

            batch_size: int
                The batch size for datasets
                Defaults to 10.

            iris_dataset_csv_path: str
                String path to the iris dataset csv file

            **kwargs
                Other keyword arguments to be passed to super()
                Anything in this set is optional for this task 
                e.g. optional GenericTask parameters
        """

        self.feature_column_names = feature_column_names
        self.original_dataframe = pd.read_csv(iris_dataset_csv_path)
        self.training_batches = training_batches
        self.validation_batches = validation_batches
        self.batch_size = batch_size

        (train_dataset, validation_dataset) = self.create_datasets()
        super().__init__(
            name = name,
            model = model,
            model_base_loss = model_base_loss,
            training_dataset=train_dataset,
            training_batches = training_batches,
            validation_dataset=validation_dataset,
            validation_batches = validation_batches,
            **kwargs)

    def create_datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Creates (and returns) a tuple of (training_dataset, validation_dataset)
        based on the Iris dataset
        """

        # Here we define how many items we need from the training samples
        # Then group the feature columns by class (to evenly sample)
        # Before finally taking exactly the indices corresponding to
        # (0,1,...,training_samples) using np.arange
        # Note we divide by 3 because we know there are 3 classes
        training_samples = self.training_batches * self.batch_size
        training_dataframe = self.original_dataframe \
            .groupby("class") \
            .apply(lambda x: x.take(
                np.arange(training_samples/3)  # type: ignore
            ))
        # Repeat the same with validation but this time offset indices by
        # already consumed training_samples
        validation_samples = self.validation_batches * self.batch_size
        validation_dataframe = self.original_dataframe \
            .groupby("class") \
            .apply(lambda x: x.take(
                np.arange(training_samples/3, (training_samples+validation_samples)/3)  # type: ignore
            ))

        # Now each dataframe consists of unique items of evenly sampled classes,
        # We can process into tensorflow datasets
        training_features = training_dataframe[self.feature_column_names]
        training_labels = pd.get_dummies(training_dataframe["class"], prefix="class")
        training_dataset = tf.data.Dataset.from_tensor_slices((training_features,training_labels)) \
            .shuffle(training_samples) \
            .batch(self.batch_size)

        validation_features = validation_dataframe[self.feature_column_names]
        validation_labels = pd.get_dummies(validation_dataframe["class"], prefix="class")
        validation_dataset = tf.data.Dataset.from_tensor_slices((validation_features, validation_labels)) \
            .shuffle(validation_samples) \
            .batch(self.batch_size)
        
        return (training_dataset, validation_dataset)
