import cv2
import numpy as np
from PIL import ImageGrab
import pygame
from vjoy import vj, setJoy

from keras.models import load_model

def get_region_of_interest(image, vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    result = cv2.bitwise_and(image, mask)
    
    return result

def draw_lines(image, lines):
    if lines is None:
        return

    for line in lines:
        temp = line[0]
        color = [255, 0, 0]
        cv2.line(image, (temp[0], temp[1]), (temp[2], temp[3]), color, 5)


def process_img(original_img):
    img_gs = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    img_canny = cv2.Canny(img_gs, 50, 100)

    roi_vertices = np.array([[35, 405], [155, 225], [240, 220], [465, 405]])
    img_canny = get_region_of_interest(img_canny, [roi_vertices])
    img_canny = cv2.GaussianBlur(img_canny, (5, 5), 0)

    lines = cv2.HoughLinesP(img_canny, 4, np.pi/180, 180, 10, 20)    
    draw_lines(img_canny, lines)

    return img_canny

def steer(angle):
    # TODO: Create algorith for smooth steering
    setJoy(angle, 0, 16000)

if __name__ == "__main__":

    model = load_model("../models/autopilot2.h5")

    vj.open()
    setJoy(0, 0, 16000)

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

        steering_angle = 0
        # TODO: Normalization ?
        steering_angle = predicted_angle

        print("Predicted: " + str(predicted_angle) + " Steering angle: " + str(steering_angle))

        steer(steering_angle)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    # Reset and close virtual joystick
    setJoy(0, 0, 16000)
    vj.close()