import numpy as np
import pandas as pd
import cv2
import datetime

from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, BatchNormalization, Conv2D, Lambda, Dropout
from keras.losses import mean_squared_error
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

# PilotNet CNN architecture
def create_cnn(shape):
    return Sequential([
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

def main():
    print("Loading dataset...")

    data = pd.read_csv("./dataset3/dataset3.csv")

    X = []
    X_flipped = []

    for img_name in data.image:
        image = cv2.imread("./dataset3/" + img_name)
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
    print("Number of images: " + str(len(X)))

    shape = X[0].shape

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    X = np.array(X)
    y = np.array(y)

    model = create_cnn(shape)

    # Custom PilotNet 

    # model = Sequential([
    #     Conv2D(24, kernel_size=(5,5), strides=(2,2), activation='relu', input_shape=shape),
    #     BatchNormalization(),
    #     Conv2D(36, kernel_size=(5,5), strides=(2,2), activation='relu'),
    #     BatchNormalization(),
    #     Conv2D(48, kernel_size=(5,5), strides=(2,2), activation='relu'),
    #     BatchNormalization(),
    #     Conv2D(64, kernel_size=(3,3), strides=(1,1), activation='relu'),
    #     BatchNormalization(),
    #     Conv2D(64, kernel_size=(3,3), strides=(1,1), activation='relu'),
    #     BatchNormalization(),
    #     Flatten(),
    #     Dense(100, activation='relu'),
    #     BatchNormalization(),
    #     Dense(50, activation='relu'),
    #     BatchNormalization(),
    #     Dense(10, activation='relu'),
    #     BatchNormalization(),
    #     Dense(1)
    # ])

    model.compile(
        optimizer=Adam(),
        loss=mean_squared_error,
        metrics=['accuracy']
    )

    # Probati batch normalization ??
    # Shuffle dataset-a ?

    model.fit(X, y, batch_size=64, validation_split=0.2, epochs=2, shuffle=True)
    model.save("./models/autopilot.h5")

    # model.fit(np.array(train_X),np.array(train_Y),
    #       batch_size=32,nb_epoch=20,
    #       validation_data=(np.array(valid_X),np.array(valid_Y)),
    #       callbacks=[early_stop])

    # How to make a prediction
    # angle = model.predict(np.expand_dims(X[0], axis=0))
    # print(angle)

if __name__ == "__main__":
    main()