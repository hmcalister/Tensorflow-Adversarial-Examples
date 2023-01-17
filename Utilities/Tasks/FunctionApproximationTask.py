# fmt: off
from typing import Callable, Tuple
import numpy as np
from .GenericTask import GenericTask

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
# fmt: on

class FunctionApproximationTask(GenericTask):
    """
    A task about modelling a function that maps inputs (independent variables) to outputs
    Functions are the data functions, and (if independent variables != model_input_shape)
    input_data_function maps independent variables to input tensors
    """

    def __init__(self, 
            name: str, 
            model: tf.keras.models.Model, 
            model_base_loss: tf.keras.losses.Loss,
            independent_variables: int,
            model_input_shape: Tuple[int,],
            input_data_fn: Callable,
            data_fn: Callable,
            training_batches: int = 0,
            validation_batches: int = 0,
            batch_size: int = 32,
            x_lim: Tuple[float, float] = (-1,1),
            **kwargs,
        ) -> None:
        """
        Create a new FunctionApproximationTask.

        Parameters:
            name: str
                The name of this task. Usually like "Task 1"

            model: tf.keras.models.Model
                The model to fit to the tasks data

            model_base_loss: tf.keras.losses.Loss:
                The base loss function of the model (before EWC)
            
            independent_variables: int
                The number of independent variables to create for each data

            model_input_shape: Tuple[int,]
                The input shape of the model

            input_data_fn: function
                The function to map independent variables to model inputs
                (If no mapping required, use lambda x: x)

            data_fn: function
                The function to map independent variables to model outputs

            training_batches: int
                The number of batches in the training dataset
            
            validation_batches: int
                The number of batches in the validation dataset

            batch_size: int
                The batch size for datasets
                Defaults to 32.

            x_lim: Tuple[float, float]:
                The input limits to the data function
                Defaults to (-1,1)

            **kwargs
                Other keyword arguments to be passed to super()
                Anything in this set is optional for this task 
                e.g. optional GenericTask parameters
        """

        self.model_input_shape = model_input_shape
        self.independent_variables = independent_variables
        self.batch_size = batch_size
        self.training_samples = batch_size
        self.training_samples = training_batches * batch_size
        self.validation_samples = validation_batches * batch_size

        self.input_data_fn = input_data_fn
        self.data_fn = data_fn
        self.x_lim = x_lim

        super().__init__(
            name = name,
            model = model,
            model_base_loss = model_base_loss,
            training_dataset=self.create_dataset(training_batches * batch_size),
            training_batches = training_batches,
            validation_dataset=self.create_dataset(validation_batches * batch_size),
            validation_batches = validation_batches,
            **kwargs)

        # Set typing correctly
        self.x_lim: Tuple[float, float]
        self.input_data_fn: Callable
        self.data_fn: Callable

    def data_generator(self, max_samples):
        i = 0
        while i < max_samples:
            x = np.random.uniform(self.x_lim[0], self.x_lim[1], self.independent_variables)
            y = self.data_fn(x)
            yield self.input_data_fn(x), y
            i += 1

    def create_dataset(self, max_samples):
        return tf.data.Dataset.from_generator(
            self.data_generator,
            args=[max_samples],
            output_signature=(
                tf.TensorSpec(shape=self.model_input_shape, dtype=tf.float64),  # type: ignore
                tf.TensorSpec(shape=(), dtype=tf.float64),  # type: ignore
            )).batch(self.batch_size).repeat()
