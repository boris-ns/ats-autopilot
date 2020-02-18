import numpy as np
import pandas as pd
import cv2
import datetime
import sys
import os

from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Activation, BatchNormalization, Conv2D, Lambda, Dropout
from keras.losses import mean_squared_error
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

# PilotNet CNN architecture
def create_cnn(shape):
    model = Sequential([
        Conv2D(24, kernel_size=(5,5), strides=(2,2), activation='relu', input_shape=shape),
        Conv2D(36, kernel_size=(5,5), strides=(2,2), activation='relu'),
        Conv2D(48, kernel_size=(5,5), strides=(2,2), activation='relu'),
        Conv2D(64, kernel_size=(3,3), strides=(1,1), activation='relu'),
        Conv2D(64, kernel_size=(3,3), strides=(1,1), activation='relu'),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(50, activation='relu'),
        Dense(10, activation='relu'),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(),
        loss=mean_squared_error,
        metrics=['accuracy']
    )

    return model

def main():
    model_path = "../models/autopilot_new.h5"
    dataset_paths = [
        "../dataset4/",
        "../dataset3/",
        "../dataset5/",
        "../dataset6/",
    ]
    
    model = None

    for dataset_path in dataset_paths:
        print("\nLoading dataset from '" + dataset_path + "'...")

        dataset_csv_path = os.path.join(dataset_path, 'data.csv')
        data = pd.read_csv(dataset_csv_path)

        X = []
        y = []

        for img_name in data.image:
            image = cv2.imread(os.path.join(dataset_path, img_name))
            
            img_gs = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            img_canny = cv2.Canny(img_gs, 50, 100)
            img_canny = cv2.GaussianBlur(img_canny, (5, 5), 0)

            # X.append(np.expand_dims(img_gs, axis=1))
            X.append(img_canny[:, :, np.newaxis])

        for angle in data.angle:
            y.append(angle)

        shape = X[0].shape

        print("\nDataset '" + dataset_path + "' loaded")
        print("Number of images: " + str(len(X)) + "\n")
        print("Shape: " + str(shape))

        if model == None:
            model = create_cnn(shape)

        X = np.array(X)
        y = np.array(y)

        print("\nStarted training for dataset '" + dataset_path + "'")
        model.fit(X, y, batch_size=64, validation_split=0.2, epochs=5, shuffle=True)
        print("\nFinished training for dataset '" + dataset_path + "'")


    model.save(model_path)
    print("\nTraining finished")
    print("Model is saved to this path: " + model_path)
    
    # How to make a prediction
    # angle = model.predict(np.expand_dims(X[0], axis=0))
    # print(angle)

if __name__ == "__main__":
    main()