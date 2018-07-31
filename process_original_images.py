"""
This script converts the original images into face images using OpenCV.
"""

import numpy as np
import cv2
import os

FACE_SIZE = 40

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

ORIGINAL_DATA_DIR = 'original_data'
PROCESSED_DATA_DIR = 'processed_data'
exclude_list = {'.gif', '.svg'}



def show_single_image(image_path, output_path):
    print('Processing ' + image_path)
    ext = image_path[image_path.rfind('.'):]
    if ext in exclude_list:
        print('Skipped')
        return
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# return first
def get_first_face_image(image_path):
    img = cv2.imread(image_path)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.imread(image_path, 0)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        #face_img = cv2.GetSubRect(x, y, w, h)
        face_img = img[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (FACE_SIZE, FACE_SIZE))
        return face_img
    return None


def process_single_image(image_path, output_path_formatter):
    print('Processing ' + image_path)
    ext = image_path[image_path.rfind('.'):]
    if ext in exclude_list:
        print('Skipped')
        return
    if os.path.exists(output_path_formatter.format(1)):
        print('Processed. Skipped.')
        return
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    index = 1
    for (x, y, w, h) in faces:
        #face_img = cv2.GetSubRect(x, y, w, h)
        face_img = img[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (FACE_SIZE, FACE_SIZE))
        out_path = output_path_formatter.format(index)
        #face_img.save(out_path)
        print(out_path)
        cv2.imwrite(out_path, face_img)
        index += 1

def process_directory(directory_path, class_name):
    target_dir = os.path.join(PROCESSED_DATA_DIR, class_name)
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    index = 1
    for root, dirs, files in os.walk(directory_path):
        for f in files:
            image_path = os.path.join(directory_path, f)
            output_path = os.path.join(target_dir, str(index)+'.{}.'+f)
            try:
                process_single_image(image_path, output_path)
            except Exception as e:
                print(e)
                print('error..skipped..')
            index += 1

def process_all(original_data_dir):
    for root, dirs, files in os.walk(original_data_dir):
        for directory in dirs:
            directory_path = os.path.join(root, directory)
            class_name = directory
            process_directory(directory_path, class_name)
        # only process the directories under root, not recursively
        break

def main():
    process_all(ORIGINAL_DATA_DIR)

if __name__ == '__main__':
    main()
