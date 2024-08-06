import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load CSV files
trainData = pd.read_csv('C:/Users/Snow/Kaggle/Numbers/data/train.csv')
testData = pd.read_csv('C:/Users/Snow/Kaggle/Numbers/data/test.csv')

# Separate features and labels
x_train = trainData.drop('label', axis=1).values
y_train = trainData['label'].values

if 'label' in testData.columns:
    x_test = testData.drop('label', axis=1).values
    y_test = testData['label'].values
else:
    x_test = testData.values
    y_test = None

# Normalize the data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape the data to fit the model input requirements
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Function to visualize sample images
def plot_sample_images(x, y, num_samples=10):
    plt.figure(figsize=(10, 1))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(x[i].squeeze(), cmap='gray')
        plt.title(y[i] if y is not None else "No label")
        plt.axis('off')
    plt.show()

# Visualize some sample images
plot_sample_images(x_train, y_train)
