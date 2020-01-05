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
    # Default path
    dataset_path = "../dataset3/" 

    continue_training = False
    if len(sys.argv) == 2:
        continue_training = True
        dataset_path = sys.argv[1]

    print("\nLoading dataset from '" + dataset_path + "'...")

    dataset_csv_path = os.path.join(dataset_path, 'data.csv')
    data = pd.read_csv(dataset_csv_path)

    X = []
    X_flipped = []

    for img_name in data.image:
        image = cv2.imread(os.path.join(dataset_path, img_name))
        X.append(image)    
    #     X_flipped.append(cv2.flip(image, 1))

    # for flipped in X_flipped:
    #     X.append(flipped)

    y = []
    y_flipped = []

    for angle in data.angle:
        y.append(angle)
        # y_flipped.append(-angle)

    y += y_flipped

    # y_np_flipped = np.array(y_flipped)

    print("\nDataset loaded")
    print("Number of images: " + str(len(X)) + "\n")

    shape = X[0].shape

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    X = np.array(X)
    y = np.array(y)

    model = None

    if continue_training:
        model = load_model("../models/autopilot.h5")
        print("\nLoaded existing model")
    else:
        model = create_cnn(shape)
        print("\nCreated new model")

    print("\nStarted training...\n")

    model.fit(X, y, batch_size=64, validation_split=0.2, epochs=8, shuffle=True)
    model.save("../models/autopilot.h5")

    print("\nTraining finished")

    # How to make a prediction
    # angle = model.predict(np.expand_dims(X[0], axis=0))
    # print(angle)

if __name__ == "__main__":
    main()