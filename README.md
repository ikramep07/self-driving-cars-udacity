
# self-driving-car-simulation using Udacity environment
Self driving car simulation using Udacity.The goal of this project is to make a car drive autonomously using a deep learning approach by feeding an input image to a neural network and predicting the steering angle.

## Code structure: 
* (`main.py`) : Training code.
* (`drive.py`):Testing code, it contains the script to connect to the simulator and run your model.


## Dataset:
The dataset is collected from the udacity self driving car simulator or downloaded from kaggle.

## Dataset directory structure:
```
Data directory (data)
├── driving_log.csv
└─┬ IMG
  └── center_2019_02_12_17_11_35_312.jpg
  └── .....
```

## Install dependecies:
```
pip install -r dependecies.txt
```

## Train on your own dataset:
run the command to see the available arguments:
```
python main.py 
```
### Test the Model: 
```
 python drive.py
```
### the pretrained model: 
```
 sfd_model.H5
```
