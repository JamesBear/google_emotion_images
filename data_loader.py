"""
This program compress image data into a pickle file and load the pickle file.
"""
import numpy as np
import cv2
import os

DATA_DIR = 'processed_data'
FACE_SIZE = 40

def load_directory(directory_path):
    count = 0
    for root, dirs, files in os.walk(directory_path):
        for f in files:
            count += 1
        break
    X = np.zeros([count, FACE_SIZE, FACE_SIZE, 3])
    index = 0
    for root, dirs, files in os.walk(directory_path):
        for f in files:
            image_path = os.path.join(directory_path, f)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            X[index,:,:,:] = img/255
            #cv2.imshow('random pic', X[index])
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            index += 1

        break
    return X, count


def empty(x):
    return type(x) == type(None)

def load_all_images(data_dir):
    X = None
    y = []
    for root, dirs, files in os.walk(data_dir):
        for directory in dirs:
            directory_path = os.path.join(root, directory)
            class_name = directory
            data, count = load_directory(directory_path)
            if empty(X):
                X = data
            else:
                X = np.r_[X, data]
            y += [class_name]*count
        # only process the directories under root, not recursively
        break
    y = np.array(y)
    return X, y

def load_data():
    X, y = load_all_images(DATA_DIR)

    print('X.shape = ', X.shape)
    print('y.shape = ', y.shape)
    #cv2.imshow('random pic', X[1099, :, :, :])
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return X, y
