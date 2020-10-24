
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad
import argparse
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import numpy as np
from random import shuffle
import random
import matplotlib.pyplot as mpimg
from imgaug import augmenters as iaa
import cv2
import tensorflow.keras.metrics as metrics

def path_del(path):
    """
    Parameters:
        argument1 (str): the whole path to the image.
    Returns:
        image name only(str)
    """
    path = "\\".join(path.strip("\\").split('\\')[6:])
    return path

def load_data(data_dir , path_delete_function):
    """load the driving log csv file  from the directory and do some processing
    Parameters:
        argument1 (str): path to the csv file
        argument1 (function): function to delete the extra .....
    Returns:
        data as pandas data frame object.
    """
    columns=  ['center' , 'left' , 'right' , 'steering' , 'throttle' , 'reverse' , 'speed']
    data = pd.read_csv(os.path.join(data_dir , 'driving_log.csv') , names = columns)
    pd.set_option('display.max_colwidth' , -1)
    data['center'] = data['center'][1].split('\\')[-1]
    data['right'] = data['right'][1].split('\\')[-1]
    data['left'] = data['left'][1].split('\\')[-1]
    return data

def load_img_steering(datadir,data):
    """load the images paths with the correspending steering angle
    Parameters:
        argument1 (str): the whole path to the image.
        argument2 (function): function to delete the extra .....
    Returns:
        img_paths(ndarray): path to images as numpy array.
        steerings(ndarray): the correspending steering angle.
    """
    img_path=[]
    steering = []
    for i in range(len(data)):
        indexed_data = data.iloc[i]
        center , left , right = indexed_data[0] , indexed_data[1] , indexed_data[2]
        img_path.append(os.path.join(datadir , center.strip()))
        steering.append(float(indexed_data[3]))
    img_paths = np.asarray(img_path)
    steerings = np.asarray(steering)
    return img_paths , steerings


def delete_useless_data(data , nb_bins = 25 , samples_per_bins = 200):
    """balancing ground truth distribution to steering = 0
    Parameters:
        argument1 (DataFrame): the whole path to the image.
        argument2 (int): divide the entire range of values into a series of intervals of 25
        argument3 (int): number of samples per bins

    Returns:
        data as pandas DataFrame.
    """
    remove_list = []
    hist , bins= np.histogram(data['steering'] , nb_bins)
    for j in range(nb_bins):
        list_ex = []
        for i in range(len(data['steering'])):
            if data['steering'][i] >= bins[j] and data['steering'][i]<= bins[j+1]:
                list_ex.append(i)
        list_ex = list(np.random.permutation(list_ex))
        #list_ex = shuffle(list_ex)
        list_ex = list_ex[samples_per_bins:]
        remove_list.extend(list_ex)
    data.drop(data.index[remove_list], inplace=True)
    return data

""" bunch of augmentation functions"""
def zoom(img):
    zoom = iaa.Affine(scale=(1,1.3))
    img = zoom.augment_image(img)
    return img

def pan(img):
    pan = iaa.Affine(translate_percent={"x" : (-0.1,0.1)  , "y" : (-0.1,0.1)  })
    img = pan.augment_image(img)
    return img

def brightness(img):
    brightness = iaa.Multiply((0.2,1.2))
    img = brightness.augment_image(img)
    return img

def flip(img,steering_angle):
    img = cv2.flip(img , 1 )
    steering_angle = - steering_angle

    return img , steering_angle

def random_augument(img, steering_angle):
    """randomly augment the dataset
    Parameters:
        argument1 (numpy array): the whole path to the image.
        argument2 (float): divide the entire range of values into a series of intervals of 25


    Returns:
        augmented image and the correspending steering angle
    """
    
    img = mpimg.imread(img)
    if np.random.rand() < 0.5:
        img = zoom(img)

    if np.random.rand() < 0.5:
        img = brightness(img)

    if np.random.rand() < 0.5:
        img = pan(img)

    if np.random.rand() < 0.5:
        img , steering_angle = flip(img, steering_angle)

    return img , steering_angle

def img_preprocess(img):
    """preprocessing the image before feeding it to the model
    Parameters:
        argument1 (numpy array): the whole path to the image.
        argument2 (float): divide the entire range of values into a series of intervals of 25


    Returns:
        the preprocessed image.
    """

    img = img[60:135:,:]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img = cv2.GaussianBlur(img,(3,3), 0)
    img = cv2.resize(img,(200,66))
    img = img/255
    return img


def batch_generator(img_paths,steerings, istraining,  batch_size):
    """Generator that yield batches of training data
    Parameters:
        argument1 (str):  path to the image.
        argument2 (float): correspending steering.
        argument2 (int): batch size.
        argument2 (bool): trainbing or validation.

    Returns:
        the batch of the input frame an the batch steering angle
    """
    while True:
        batch_img = []
        batch_steering = []
        for i in range(batch_size):
            random_index = random.randint(0, len(img_paths)-1)
            if istraining:
                im , steering = random_augument(img_paths[random_index] , steerings[random_index])
            else:
                im = mpimg.imread(img_paths[random_index])
                steering = steerings[random_index]

            im = img_preprocess(im)
            batch_img.append(im)
            batch_steering.append(steering)

        yield np.asarray(batch_img) , np.asarray(batch_steering)

def visualize_model_graph(history , is_accuracy):
    """Generator that yield batches of training data
    Parameters:
        argument1 (str):  path to the image.
        argument2 (bool): plotting accuracy or loss

    Returns:
        None
    """
    if is_accuracy:
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.legend(['training',' validation'])
        plt.title('accuracy')
        plt.xlabel('Epoch')
        plt.show()
    else:
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.legend(['training',' validation'])
        plt.title('Loss')
        plt.xlabel('Epoch')

def sfd_model(optimizer, learning_rate):
    '''
    nvidia self driving car inspired architecture.
    '''

    if optimizer == 'adagrad':
        optimizer = Adagrad(lr=learning_rate)
    elif optimizer == 'sgd':
        optimizer = SGD(lr=learning_rate)
    elif optimizer == 'rmsprop':
        optimizer = RMSprop(lr=learning_rate)
    else:
        optimizer = Adam(lr=learning_rate)

    model = Sequential()
    model.add(Conv2D(24, 5, 2, input_shape=(66, 200, 3), activation='elu'))
    model.add(Conv2D(36, 5, 2,  activation='elu'))
    model.add(Conv2D(48, 5, 2,  activation='elu'))
    model.add(Conv2D(64, 3, activation='elu'))
    model.add(Conv2D(64, 3, activation='elu'))
    # model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='elu'))
    # model.add(Dropout(0.5))
    model.add(Dense(64, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    
    model.compile(loss='mse', optimizer=optimizer, metrics= metrics.Accuracy(name='Accuracy'))
    return model

parser = argparse.ArgumentParser(description='command line for diffrent parameters')
parser.add_argument('--dir', type = str , default ='data', help='data directory for the csv and the images')
parser.add_argument('--optimizer', type = str ,required = False , default = 'adam', help='possible arguments : sgd , adagrad , adam , rmsprop. default: Adam')
parser.add_argument('--lr', type = float ,required = False , default = 0.0001,
                    help='learning rate for the optimizer ex : 0.01, 0,001 . default : 0.001')
parser.add_argument('--batch_size', type = int ,required = False , default = 32 ,
                    help='how many samples per batch. default : 32')
parser.add_argument('--epochs', type = int ,required = False , default = 25 ,
                    help='how many samples per batch. default : 32')

args = parser.parse_args()
data_dir =  args.dir
print(' data foldzer is :', data_dir)
#hyperparameters
optimizer = args.optimizer
lr = args.lr
batch_size = args.batch_size
nb_epchs = args.epochs


#load the driving log data
data = load_data(data_dir, path_del)

#balancing the ground truth data distribution to steering = 0
new_data = delete_useless_data(data)
#loading images along with the correspending steering angles
img_paths , steerings = load_img_steering(data_dir + '\IMG' , new_data)

#training and validation split
X_train , X_val , y_train  , y_val = train_test_split(img_paths , steerings , test_size=0.2 , random_state = 6)
print('Training samples : {}\nValidation samples : {}'.format(len(X_train) , len(X_val)) )
print(y_train.shape, y_val.shape)
#create batch generated data for efficiency.
X_train_gen , y_train_gen = next(batch_generator(X_train, y_train, 1 ,  batch_size = batch_size))
X_valid_gen , y_val_gen = next(batch_generator(X_val, y_val,0 ,batch_size = batch_size))

model = sfd_model(optimizer = optimizer  , learning_rate = args.lr)

#train the model on data generated batch by batch
history = model.fit_generator(batch_generator(X_train , y_train ,1 , batch_size = batch_size) ,
                              steps_per_epoch = 300,
                              epochs = nb_epchs,
                              validation_data = batch_generator(X_val , y_val ,0,batch_size = batch_size) ,
                              validation_steps = 200,
                              verbose = 1 ,
                              shuffle = 1)


#Save the model
model.save('model_b.h5')