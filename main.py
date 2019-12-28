import cv2
import numpy as np
from PIL import ImageGrab
import pygame
from vjoy import vj, setJoy

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

if __name__ == "__main__":

    pygame.joystick.init()
    joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]

    print("Number of joysticks: {0}".format(pygame.joystick.get_count()))

    # TODO: provera da li je detektovan dzojstik

    pygame.display.init()

    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(joystick.get_name())
    print(joystick.get_numaxes())

    vj.open()

    while(True):
        setJoy(1, 0, 16000)

        img = ImageGrab.grab(bbox=(500,100,1000,510)) # (bbox= x,y,width,height)
        img_np = np.array(img)
        frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        processed_img = process_img(frame)

        cv2.imshow("Processed image", processed_img)
        
        # Middle line
        cv2.line(frame, (200, 0), (200, 700), [255, 255, 0], 3)
        
        # Horizontal line
        cv2.line(frame, (0, 350), (500, 350), [255, 0, 0], 3)

        cv2.imshow("Original", frame)

        pygame.event.pump()
        print("X axis: {0}".format(joystick.get_axis(0)))

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    vj.close()