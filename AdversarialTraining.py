# fmt: off
import pandas as pd
from Utilities.Utils import *
from Utilities.Tasks.CIFAR10ClassificationTask import CIFAR10ClassificationTask as Task
from Utilities.AdversarialTraining import AdversarialExampleTrainer

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
# fmt: on
print(f"GPU: {tf.config.list_physical_devices('GPU')}")

# Define paths to save important data to 
MODEL_SAVE_PATH = "models/CIFAR10_INTERLEAVED_ADVERSARIAL_MODEL"
HISTORY_SAVE_PATH = "history.csv"
# True for easier debugging
# False for compiled models, faster train time
RUN_EAGERLY: bool = False

# Training parameters
ITERATION_MAX = 100
VANILLA_EPOCHS = 10
ADVERSARIAL_EPOCHS = 1
training_batches = 0
validation_batches = 0
batch_size = 32
model_input_shape = Task.IMAGE_SIZE

# Labels to classify in each task
task_labels = [
    i for i in range(10)
]

# Create a model for the task
model_inputs = model_layer = tf.keras.Input(shape=model_input_shape)
model_layer = tf.keras.layers.Conv2D(32, (3,3), activation="relu", name="conv2d_0")(model_layer)
model_layer = tf.keras.layers.Conv2D(32, (3,3), activation="relu", name="conv2d_1")(model_layer)
model_layer = tf.keras.layers.Conv2D(32, (3,3), activation="relu", name="conv2d_2")(model_layer)
model_layer = tf.keras.layers.MaxPool2D((2,2))(model_layer)
model_layer = tf.keras.layers.BatchNormalization()(model_layer)
model_layer = tf.keras.layers.Conv2D(64, (3,3), activation="relu", name="conv2d_3")(model_layer)
model_layer = tf.keras.layers.Conv2D(64, (3,3), activation="relu", name="conv2d_4")(model_layer)
model_layer = tf.keras.layers.MaxPool2D((2,2))(model_layer)
model_layer = tf.keras.layers.BatchNormalization()(model_layer)
model_layer = tf.keras.layers.Flatten()(model_layer)
model_layer = tf.keras.layers.Dense(64, activation="relu")(model_layer) 
model_layer = tf.keras.layers.Dense(64, activation="relu")(model_layer)
model_layer = tf.keras.layers.Dense(len(task_labels))(model_layer)
model = tf.keras.Model(inputs=model_inputs, outputs=model_layer, name="model")

# Define a loss function for the task
# Note, this is the loss function used directly for the adversarial examples!
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

print(f"BASE MODEL SUMMARY")
model.summary()

# Introduce any image augmentation - in case that is useful
training_image_augmentation = None
training_image_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomZoom(
            height_factor=(-0.05, -0.25),
            width_factor=(-0.05, -0.25)),
    tf.keras.layers.RandomRotation(0.15)
])

# Create a sequential task - for easy of training later
task = Task(
        name="Task",
        model=model,
        model_base_loss=loss_fn,
        task_labels=task_labels,
        training_batches = training_batches,
        validation_batches = validation_batches,
        batch_size=batch_size,
        training_image_augmentation = training_image_augmentation,
        run_eagerly = RUN_EAGERLY,
    )

# Set how much perturbation is added to adversarial examples 
EPSILON = 0.01
adversarial_example_trainer = AdversarialExampleTrainer(task)

# Dataframe to store training history between task and adversarial trainer
all_history = pd.DataFrame()
try:
    for iteration_index in range(1,ITERATION_MAX):
        print(f"{'-='*80}")
        print(f"{iteration_index=}")
        print(f"{'-='*80}")

        history = task.train_on_task(epochs=VANILLA_EPOCHS)
        history = pd.DataFrame(history.history)
        history["TrainType"] = "Vanilla"
        all_history = pd.concat([all_history, history], ignore_index=True)
        all_history.to_csv(HISTORY_SAVE_PATH)
        model.save(MODEL_SAVE_PATH)        

        history = adversarial_example_trainer.train_adversarial(epsilon=EPSILON, epochs=ADVERSARIAL_EPOCHS)
        history = pd.DataFrame(history.history)
        history["TrainType"] = "Adversarial"
        all_history = pd.concat([all_history, history], ignore_index=True)
        all_history.to_csv(HISTORY_SAVE_PATH)
        model.save(MODEL_SAVE_PATH)        
except KeyboardInterrupt:
    print("KEYBOARD INTERRUPT")

model.save(MODEL_SAVE_PATH)
plt.plot(all_history["loss"], label="loss")
plt.plot(all_history["val_loss"], label="val_loss")
plt.title("Interleaved Adversarial Training")
plt.ylabel("Categorical Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()
