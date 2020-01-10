from keras.models import load_model
import numpy as np
import os
import cv2
import predictFrame

model = load_model("model_newCorn.h5")

dataTrain = np.genfromtxt('video_train.csv', delimiter=',', dtype='str')

videoTrain = dataTrain[1:,0]
averages4Train = np.zeros_like(videoTrain, dtype='int32')
averages2Train = np.zeros_like(videoTrain, dtype='int32')

length4Train = list()
length2Train = list()



for video  in os.listdir('video5'):
    print(video)
    vidcap = cv2.VideoCapture('video5/' + video)
    success,image = vidcap.read()
    count = 100
    frame = 0
    success = True
    while success:
        success,image = vidcap.read()
        if(count > 112):
            cv2.imwrite("videoFrame/frame%d.jpg" % frame, image)
            frame += 1
            count = 0
        count += 1
        
    numImages = 0
    numY = 0
    for image in os.listdir('videoFrame'):
        numImages += 1
        im = cv2.imread('videoFrame/' + image)
        numY += predictFrame.predictAverage(im ,model)
        
    print(numY)
    print(numImages)
    print(numY/numImages)
    length4Train.append(numY/numImages)
    
    for image in os.listdir('videoFrame'):
        if image.endswith('.jpg'):
            os.unlink('videoFrame/' + image)
            
for i in range(len(length4Train)):
    averages4Train[i] = length4Train[i]
    
np.savetxt('averages4Train.csv', averages4Train)
    
for video  in os.listdir('video6'):
    print(video)
    vidcap = cv2.VideoCapture('video6/' + video)
    success,image = vidcap.read()
    count = 100
    frame = 0
    success = True
    while success:
        success,image = vidcap.read()
        if(count > 112):
            cv2.imwrite("videoFrame/frame%d.jpg" % frame, image)
            frame += 1
            count = 0
        count += 1
        
    numImages = 0
    numY = 0
    for image in os.listdir('videoFrame'):
        numImages += 1
        im = cv2.imread('videoFrame/' + image)
        numY += predictFrame.predictAverage(im ,model)
        
    print(numY)
    print(numImages)
    print(numY/numImages)
    length2Train.append(numY/numImages)
    
    for image in os.listdir('videoFrame'):
        if image.endswith('.jpg'):
            os.unlink('videoFrame/' + image)
            
for i in range(len(length2Train)):
    averages2Train[i] = length2Train[i]
    
np.savetxt('averages2Train.csv', averages2Train)


"""   Start Test Data   """
                                
dataTest = np.genfromtxt('video_test.csv', delimiter=',', dtype='str')

videoTest = dataTest[1:,0]
averages4Test = np.zeros_like(videoTest, dtype='int32')
averages2Test = np.zeros_like(videoTest, dtype='int32')

length4Test = list()
length2Test= list()

for video  in os.listdir('video7'):
    print(video)
    vidcap = cv2.VideoCapture('video7/' + video)
    success,image = vidcap.read()
    count = 100
    frame = 0
    success = True
    while success:
        success,image = vidcap.read()
        if(count > 112):
            cv2.imwrite("videoFrame/frame%d.jpg" % frame, image)
            frame += 1
            count = 0
        count += 1
        
    numImages = 0
    numY = 0
    for image in os.listdir('videoFrame'):
        numImages += 1
        im = cv2.imread('videoFrame/' + image)
        numY += predictFrame.predictAverage(im ,model)
        
    print(numY)
    print(numImages)
    print(numY/numImages)
    length4Test.append(numY/numImages)
    
    for image in os.listdir('videoFrame'):
        if image.endswith('.jpg'):
            os.unlink('videoFrame/' + image)
            
for i in range(len(length4Test)):
    averages4Test[i] = length4Test[i]
    
np.savetxt('averages4Test.csv', averages4Test)

for video  in os.listdir('video8'):
    print(video)
    vidcap = cv2.VideoCapture('video8/' + video)
    success,image = vidcap.read()
    count = 100
    frame = 0
    success = True
    while success:
        success,image = vidcap.read()
        if(count > 112):
            cv2.imwrite("videoFrame/frame%d.jpg" % frame, image)
            frame += 1
            count = 0
        count += 1
        
    numImages = 0
    numY = 0
    for image in os.listdir('videoFrame'):
        numImages += 1
        im = cv2.imread('videoFrame/' + image)
        numY += predictFrame.predictAverage(im ,model)
        
    print(numY)
    print(numImages)
    print(numY/numImages)
    length2Test.append(numY/numImages)
    
    for image in os.listdir('videoFrame'):
        if image.endswith('.jpg'):
            os.unlink('videoFrame/' + image)
            
for i in range(len(length2Test)):
    averages2Test[i] = length2Test[i]
    
np.savetxt('averages2Test.csv', averages2Test)


