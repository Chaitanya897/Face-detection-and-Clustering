'''
All of your implementation should be in this file.
'''
'''
This is the only .py file you need to submit. 
'''
'''
    Please do not use cv2.imwrite() and cv2.imshow() in this function.
    If you want to show an image for debugging, please use show_image() function in helper.py.
    Please do not save any intermediate files in your final submission.
'''
from helper import show_image

import cv2
import numpy as np
import os
import sys

import face_recognition
import matplotlib.pyplot as plt

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''
from sklearn.cluster import KMeans

def detect_faces(input_path: str) -> dict:
    result_list = []
    '''
    Your implementation.
    '''
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    path = input_path
    
    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        img = cv2.imread(f)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 6)
        for (x,y,w,h) in faces:
            result_list.append({"iname": filename, "bbox": [int(x), int(y), int(w), int(h)]})
    print(result_list)
    return result_list


'''
K: number of clusters
'''
def cluster_faces(input_path: str, K: int) -> dict:
    result_list = []
    result_list1 = []
    names_of_images = []
    temp = []
    '''
    Your implementation.
    '''
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    path = input_path
    enc =[]
    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        img = cv2.imread(f)
        names_of_images.append(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 6)
        for (x,y,w,h) in faces:
            encoding = face_recognition.face_encodings(img)
            temp.append(img)
        enc.append(encoding[0])
 
    kmeans=KMeans(n_clusters=5)
    kmeans.fit(enc)
    a=kmeans.labels_
    # print(a)
    u = np.unique(a)
    i = 0
    while i in u:
        l1, names = [], []
        for j in range(len(a)):
            if i == a[j]:
                names.append(names_of_images[j])
                l1.append(j)

        result_list1.append({"cluster":int(i), "elements":l1})   #for plotting
        result_list.append({"cluster":int(i), "elements":names})    #for clusters.json
        i += 1


# Code For Plotting

    # ii = 0
    # while ii in result_list1:
    # for ii in result_list1:
    #     f = plt.figure(figsize=(4, 4))
    #     r, c = 4, 4
    #     t = 0
    #     # while iii in ii['elements']:
    #     for iii in ii['elements']:
    #         t += 1
    #         f.add_subplot(r, c, t)
    #         plt.imshow(temp[iii])
    #         plt.axis("off")
        # iii += 1
        #plt.show()
        # ii += 1


    return result_list


'''
If you want to write your implementation in multiple functions, you can write them here. 
But remember the above 2 functions are the only functions that will be called by FaceCluster.py and FaceDetector.py.
'''
"""
Your implementation of other functions (if needed).
"""