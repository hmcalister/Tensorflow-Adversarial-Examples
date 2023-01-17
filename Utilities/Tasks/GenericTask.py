# fmt: off
from copy import deepcopy
from typing import Callable, List, Tuple, Union

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
# fmt: on

class GenericTask:
    """
    A container for a single sequential task. 
    Includes a model (already compiled, hopefully sharing weights with other tasks)
    and a tf.data.Dataset representing the task data (with optional validation data)
    """

    def __init__(self, 
            name: str,
            model: tf.keras.models.Model,
            model_base_loss: tf.keras.losses.Loss,
            training_dataset: tf.data.Dataset,
            training_batches: int,
            validation_dataset: tf.data.Dataset = None,  # type: ignore
            validation_batches: int = 0,
            batch_size: int = 0,
            run_eagerly: bool = False,
            input_data_fn: Union[Callable, None] = None,
            data_fn: Union[Callable, None] = None,
            x_lim: Union[Tuple[float, float], None] = None,
            y_lim: Union[Tuple[float, float], None] = None,
        ) -> None:
        """
        Create a new SequentialTask.
        A task consists of a model (already compiled), training data,
        and validation data to test the model. 

        Parameters:
            name: str
                The name of this task. Usually like "Task 1"

            model: tf.keras.models.Model
                The model to fit to the tasks data

            model_base_loss: tf.keras.losses.Loss:
                The base loss function of the model (before EWC)

            training_data: tf.data.Dataset
                The training data to fit to

            training_batches: int
                The number of batches in the training dataset

            validation_data: tf.data.Dataset
                The validation data to test on. Optional, if None no validation is done

            validation_batches: int
                The number of batches in the validation dataset
            
            run_eagerly: bool
                Boolean to compile model to tensorflow graph
        """

        self.name = name
        self.model = model
        self.model_base_loss = model_base_loss
        self.training_dataset = training_dataset
        self.training_batches = training_batches
        self.validation_dataset = validation_dataset
        self.validation_batches = validation_batches
        self.batch_size = batch_size
        self.run_eagerly = run_eagerly

        self.model_base_loss.name = "base_loss"
        model_base_loss_serialized: dict = deepcopy(tf.keras.losses.serialize(self.model_base_loss))  # type: ignore
        # remove objects that metric cannot understand
        model_base_loss_serialized["config"].pop("reduction", None)
        model_base_loss_serialized["config"].pop("axis", None)
        self.model_base_loss_as_metric: tf.keras.metrics.Metric = tf.keras.metrics.deserialize(model_base_loss_serialized) # type: ignore
        self.compile_model(model_base_loss)


    def compile_model(self, loss_fn: tf.keras.losses.Loss, **kwargs):
        """
        (Re)compile this tasks model with a new loss function, keeping the metrics
        """

        # Tensorflow will not save/load models if the metrics include a loss function
        # We must convert a keras loss to a keras metric to avoid this!
        # Current hack is to store the original name of the base loss and use this to
        # invoke a metric instance later... not great but hopefully it works!

        # Notice that EWC requires access to layer weights during training
        # If using numpy arrays (easier debugging)/layer.get_weights()
        # this is not possible with a compiled graph (Tensorflow restricts it)
        # So we must set run_eagerly to True to avoid compilation, or
        # we can use .weights and use tensorflow Tensors instead!
        if "optimizer" not in kwargs:
            kwargs["optimizer"] = "ADAM"
        self.model.compile(
                loss=loss_fn,
                metrics=[self.model_base_loss_as_metric],
                run_eagerly=self.run_eagerly,
                **kwargs)

    def train_on_task(self, epochs, callbacks: List[tf.keras.callbacks.Callback] = []) -> tf.keras.callbacks.History:
        """
        Train on the train dataset for a number of epochs. Use any callbacks given
        If self.validation_data is not None, validation data used.
        Returns the history of training
        """

        return self.model.fit(
            self.training_dataset,
            epochs=epochs,
            steps_per_epoch=self.training_batches,
            validation_data=self.validation_dataset,
            validation_steps=self.validation_batches,
            callbacks=callbacks
        )

    def evaluate_model(self) -> dict:
        """
        Run a single pass over the validation data, returning the metrics
        """

        if self.validation_dataset is None:
            return {}
        self.model_base_loss_as_metric.reset_state()
        # Return type of this is hinted incorrectly
        # Actual return type is dict
        print(f"EVALUATING: {self.model.name}")
        return self.model.evaluate(self.validation_dataset, 
            steps=self.validation_batches, return_dict=True)  # type: ignore




