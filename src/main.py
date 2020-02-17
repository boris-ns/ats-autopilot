import cv2
import numpy as np
import pygame
from PIL import ImageGrab
from vjoy import vj, setJoy
from keras.models import load_model

STEER_STEP = 0.005

def steer(prev_angle, angle):
    if prev_angle < angle:
        # GO RIGHT
        while prev_angle < angle:
            prev_angle += STEER_STEP

            if prev_angle >= angle:
                setJoy(angle, 0, 16000)
                break
            else:
                setJoy(prev_angle, 0, 16000)

    elif prev_angle > angle:
        # GO LEFT
        while prev_angle > angle:
            prev_angle -= STEER_STEP

            if prev_angle <= angle:
                setJoy(angle, 0, 16000)
                break
            else:
                setJoy(prev_angle, 0, 16000)
    else:
        # STAY
        setJoy(angle, 0, 16000)

if __name__ == "__main__":

    model = load_model("../models/autopilot2.h5")
    model.summary()

    vj.open()
    setJoy(0, 0, 16000)

    steering_angle = 0
    prev_angle = 0

    while(True):
        img = ImageGrab.grab(bbox=(500,330,850,500)) # (bbox= x,y,width,height)
        img_np = np.array(img)
        frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        # frame_gs = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        cv2.imshow("Original frame", frame)
        # cv2.imshow("Grayscale frame", frame_gs)

        predicted_angle = model.predict(np.expand_dims(frame, axis=0))[0][0]
        
        # Predict for grayscale image
        # predicted_angle = model.predict(np.expand_dims(frame_gs[:, :, np.newaxis], axis=0))[0][0]

        prev_angle = steering_angle
        steering_angle = predicted_angle

        print("Predicted: " + str(predicted_angle))

        steer(prev_angle, steering_angle)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    # Reset and close virtual joystick
    setJoy(0, 0, 16000)
    vj.close()