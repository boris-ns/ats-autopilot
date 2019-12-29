from PIL import ImageGrab
import cv2
import numpy as np
import pygame
import time
import sys

def main():
    if len(sys.argv) != 2:
        return
    
    try:
        img_file_counter = int(sys.argv[1])
    except ValueError:
        print("ERROR: Program requires 1 argument that must be an integer.")
        return

    pygame.joystick.init()

    print("Number of joysticks detected: {0}".format(pygame.joystick.get_count()))

    if pygame.joystick.get_count() == 0:
        print("ERROR: No joysticks detected")
        return

    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    print("Joystick 0 name: " + joystick.get_name())

    pygame.display.init()
    dataset_file = open("./dataset3/dataset3.csv", "a")

    recording = False

    print("Press Joystick button 0 to start recording")

    while True:
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

        if img_file_counter == 0:
            # time.sleep(10)
            dataset_file.write("image,angle")

        pygame.event.pump()
        if joystick.get_button(0) == 1 and recording == False:
            recording = True
            print("\nStarted recording")
            time.sleep(1)
        elif joystick.get_button(0) == 1 and recording == True:
            recording = False
            print("\nPaused recording")
            time.sleep(1)

        if not recording:
            continue

        time.sleep(0.2)

        img = ImageGrab.grab(bbox=(500,330,850,500)) # (bbox= x,y,width,height)
        img_np = np.array(img)
        frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        cv2.imshow("Original", frame)

        pygame.event.pump()
        print("X axis: {0}".format(joystick.get_axis(0)))

        img_filename = str(img_file_counter) + ".jpg"
        cv2.imwrite("./dataset3/" + img_filename, frame)
        img_file_counter += 1

        dataset_file.write("\n{0},{1}".format(img_filename, joystick.get_axis(0)))

    dataset_file.close()
    pygame.display.quit()

    print("STOPED RECORDING")

if __name__ == "__main__":
    main()