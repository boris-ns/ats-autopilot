import cv2
import numpy as np
import pygame
from PIL import ImageGrab
from vjoy import vj, setJoy
from keras.models import load_model

# Config
SCREEN_GRAB_BOX = (500,330,850,500) # (x, y, w, h)
MODEL_PATH = "../models/autopilot_canny.h5"
STEER_STEP = 0.005

def joystick_steer(angle):
    setJoy(angle, 0, 16000)

def steer(prev_angle, angle):
    if prev_angle < angle:
        # GO RIGHT
        while prev_angle < angle:
            prev_angle += STEER_STEP

            if prev_angle >= angle:
                joystick_steer(angle)
                break
            else:
                joystick_steer(prev_angle)

    elif prev_angle > angle:
        # GO LEFT
        while prev_angle > angle:
            prev_angle -= STEER_STEP

            if prev_angle <= angle:
                joystick_steer(angle)
                break
            else:
                joystick_steer(prev_angle)
    else:
        # STAY
        joystick_steer(angle)

if __name__ == "__main__":

    model = load_model(MODEL_PATH)
    model.summary()

    vj.open()
    joystick_steer(0)

    steering_angle = 0
    prev_angle = 0

    while(True):
        img = ImageGrab.grab(bbox=SCREEN_GRAB_BOX) # (x, y, w, h)
        img_np = np.array(img)
        frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        # frame_gs = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        img_gs = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        img_canny = cv2.Canny(img_gs, 50, 100)
        img_canny = cv2.GaussianBlur(img_canny, (5, 5), 0)

        cv2.imshow("Original frame", frame)
        # cv2.imshow("Grayscale frame", frame_gs)
        cv2.imshow("Canny", img_canny)

        # predicted_angle = model.predict(np.expand_dims(frame, axis=0))[0][0]
        
        # Predict for grayscale image or canny
        predicted_angle = model.predict(np.expand_dims(img_canny[:, :, np.newaxis], axis=0))[0][0]

        prev_angle = steering_angle
        steering_angle = predicted_angle

        print("Predicted: " + str(predicted_angle))

        steer(prev_angle, steering_angle)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    # Reset and close virtual joystick
    joystick_steer(0)
    vj.close()