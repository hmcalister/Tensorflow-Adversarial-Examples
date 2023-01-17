import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_FILE = "history.csv"
dataframe = pd.read_csv(DATA_FILE)

dataframe = dataframe.rename(columns={"Unnamed: 0": "epoch"})

vanilla_points = dataframe[dataframe["TrainType"]=="Vanilla"]
adversarial_points = dataframe[dataframe["TrainType"]=="Adversarial"]

plt.plot(dataframe["loss"], label="loss")
plt.plot(dataframe["val_loss"], label="val_loss")
# plt.scatter(vanilla_points["epoch"], vanilla_points["loss"], marker='x', color="k", label="Vanilla Loss")
# plt.scatter(adversarial_points["epoch"], adversarial_points["loss"], marker='o', color="k", label="Adversarial Loss")
plt.title("Interleaved Adversarial Training")
plt.ylabel("Categorical Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()
