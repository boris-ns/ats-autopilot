# Autopilot for American Truck Simulator

Simple lane tracking assistant for [*American Truck Simulator*](https://americantrucksimulator.com/).  
The goal is to keep the truck inside its lane.

_NOTE:_ Currently only works on Windows.  
_NOTE:_ This project should also work on Euro Truck Simulator 2, since those two games are based on the same engine.

## Demo

[Driving on the highway.](https://www.youtube.com/watch?v=T43E-zY8eXM&feature=youtu.be)

[Driving on the highway 2.](https://www.youtube.com/watch?v=uimxQz_ED38&feature=youtu.be)

[Driving on the country road.](https://www.youtube.com/watch?v=BTYjUCX89eI&feature=youtu.be)

[Driving on the country road 2.](https://www.youtube.com/watch?v=2qf9IKp55QQ&feature=youtu.be)

## How to use

Python 3.6 is required for this project to run.
Install [vJoy](http://vjoystick.sourceforge.net/site/index.php/download-a-install/download) and [vJoy DLL file](https://www.dll-files.com/vjoy.dll.html).

Download the project and install necessary dependencies. If you want you can create a virtual 
environment before running ```pip3 install -r requirements.txt```.

```
git clone https://github.com/boris-ns/ats-autopilot
cd ats-autopilot
pip3 install -r requirements.txt
```

This project contains 3 python scripts: 
- ```main.py``` - main script that loads trained model and makes predictions and steers the wheel
- ```train.py (new_train.py)``` - scripts for training 
- ```generate_dataset.py``` - script that is used for recording the screen and saving images and steering angle

### main.py

Inside ```main.py``` you can configure these variables according to your needs:
- ```SCREEN_GRAB_BOX = (x, y, w, h)``` - screen image grab of the road in front of the truck
- ```MODEL_PATH = "../models/autopilot.h5"``` - path to the trained model file
- ```STEER_STEP = 0.005``` - step for smooth steering, you can leave it like this (the smaller number, the smoother steering).
Also, before running this script make sure you set vJoy as controller inside the game.  
```
cd src
python main.py
```

### new_train.py

```train.py``` and ```new_train.py``` are very similar. For training it's better to use 
```new_train.py```. In ```main()``` method you need to configure ```model_path``` and ```dataset_paths``` variables. You can use multiple dataset folders, as long as all of them contain it's own ```data.csv``` file.  
Then, just run:  
```
cd src
python new_train.py
```

### generate_dataset.py

In config section set values according to your needs. 

This will start saving images from 0.
```
python src/generate_dataset.py
```
If you want you can pass an integer as command line argument, and the script will just 
continue saving images from that number into the existing dataset folder.
```
cd src
python generate_dataset.py 1234
```

The script is configured to use Joystick button 0 as a command to start or pause recording.

## Trained models

You can find two trained models [here](https://drive.google.com/open?id=1tjCDPcJwzq5sHgOkz-q3fp-0RRqaCUZI).  
- ```autopilot2.h5``` is made with 40k images dataset with no image processing.
- ```autopilot_canny.h5``` is made with 40k images dataset with canny edge detection.
