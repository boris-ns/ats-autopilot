import numpy as np
import pandas as pd
import cv2

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

from sklearn.model_selection import train_test_split

def main():
    data = pd.read_csv("./dataset/dataset.csv")

    print(data.head())

    X = []

    for img_name in data.image:
        X.append(cv2.imread("./dataset/" + img_name))    

    print("Images from dataset loaded.")

    y = data.angle

    shape = X[0].shape

    num_filters = 8
    filter_size = 3
    pool_size = 2

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    print(len(X_train))
    print(len(X_test))

    print(len(y_train))
    print(len(y_test))

    X = np.array(X)
    y = np.array(y)

    model = Sequential([
        Conv2D(num_filters, filter_size, activation="relu", input_shape=shape),
        MaxPooling2D(pool_size=pool_size),
        Flatten(),
        Dense(1, activation="softmax")
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # model.fit(np.array(train_X),np.array(train_Y),
    #       batch_size=32,nb_epoch=20,
    #       validation_data=(np.array(valid_X),np.array(valid_Y)),
    #       callbacks=[early_stop])

    model.fit(X, y, batch_size=8, validation_split=0.1, epochs=1)
    angle = model.predict(np.expand_dims(X[0], axis=0))
    print(angle)

if __name__ == "__main__":
    main()