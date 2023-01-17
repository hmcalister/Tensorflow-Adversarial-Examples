# fmt: off
from typing import List
from Utilities.Utils import plot_images
from Utilities.Tasks.GenericTask import GenericTask 

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
# fmt: on



class AdversarialExampleTrainer():
    """
    Defines a class that allows for a model to be trained adversarially
    This means images in the training dataset are fed through the network,
    gradients are calculated, and the image is perturbed to maximally trick the network
    This training can help improve network stability/performance and can help with some interpretability methods
    """

    def __init__(self, task: GenericTask) -> None:
        """
        Create a new adversarial trainer from a sequential learning task

        Parameters:
            Task : GenericTask
                The task to take information from - models, datasets, loss function...
        """
        self.model = task.model
        self.loss_fn = task.model_base_loss
        self.training_batches = task.training_batches
        self.training_dataset = task.training_dataset
        self.validation_dataset = task.validation_dataset
        self.validation_batches = task.validation_batches

    def _create_adversarial_mapping(self, epsilon=0.01):
        """
        Create and return a function to map a dataset to an adversarial dataset
        This method is neccesary as mapping functions in tensorflow need to accept only (x,y) tuples
        So any associated class method (taking self) would fail, but without self we could not use model!

        This method creates the mapping function and returns it - thanks functional programming!
        """
        def f(image, label):
            with tf.GradientTape() as tape:
                tape.watch(image)
                prediction = self.model(image)
                loss = self.loss_fn(label, prediction)

            # Get the gradients of the loss w.r.t to the input image.
            gradient = tape.gradient(loss, image)
            # Get the sign of the gradients to create the perturbation
            signed_grad = tf.sign(gradient)
            perturbations = signed_grad * epsilon
            return image + perturbations, label
        return f

    def train_adversarial(self, epsilon=0.01, epochs=1, callbacks:List[tf.keras.callbacks.Callback]=[]):
        """
        Train against an adversarial example dataset

        Parameters:
            epsilon
                The amount of perturbation to add to the images. Smaller will result in slow but more stable training
            epochs
                The number of epochs to train for
            callbacks
                Any extra callbacks to add to training
        """
        
        mapping_fn = self._create_adversarial_mapping(epsilon)
        adversarial_dataset = self.training_dataset.map(mapping_fn)
        return self.model.fit(
            adversarial_dataset,
            epochs=epochs,
            steps_per_epoch=self.training_batches,
            validation_data=self.validation_dataset,
            validation_steps=self.validation_batches,
            callbacks=callbacks
        )
        
    def display_adversarial_images(self, epsilon=0.01, num_images=16, titles=False, use_validation_dataset=False, **kwargs):
        """
        Create and display a number of adversarial images

        Parameters:
            epsilon
                The amount of perturbation to add to the images. Smaller will result in slow but more stable training
            num_images:
                The number of images to display
            title:
                Boolean - add titles to subplots. Titles contain true label, predicted labels, and confidence 
            use_validation_dataset
                Boolean to use validation data rather than training data
            **kwargs
                Any additional arguments to pass to plot_images
        """

        mapping_fn = self._create_adversarial_mapping(epsilon)
        if use_validation_dataset:
            target_dataset = self.validation_dataset
        else:
            target_dataset = self.training_dataset
        vanilla_dataset = target_dataset.unbatch().take(num_images).batch(num_images)
        perturbed_dataset = vanilla_dataset.map(mapping_fn)

        perturbed_images, labels =perturbed_dataset.take(1).get_single_element()
        labels = tf.argmax(labels, axis=1)

        subplot_titles = []
        if titles:
            perturbed_predictions = self.model(perturbed_images)
            perturbed_confidences = tf.reduce_max(tf.nn.softmax(perturbed_predictions),axis=1)
            perturbed_predictions = tf.argmax(perturbed_predictions,axis=1)
            for label, prediction, confidence in zip(labels, perturbed_predictions, perturbed_confidences):
                subplot_titles.append(f"True: {label}\nPred:{prediction} ({confidence:.3f})")
        
        plot_images(perturbed_images, subplot_titles=subplot_titles, **kwargs) # type: ignore