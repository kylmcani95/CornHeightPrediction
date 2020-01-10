

import numpy as np

import cv2

def imagePyramid(image, scale=1.5, minDim=(250, 140)):
    yield image
    
    while True:
        width = int(image.shape[1]/ scale)
        height = int(image.shape[0]/scale)
        dim = (width, height)
        image = cv2.resize(image, dim)
        
        if image.shape[0] < minDim[1] or image.shape[1] < minDim[0]:
            break
        
        yield image


def boundingBox(image, imHeight, imWidth, stepSize, boxSize):
    for height in range(0, imHeight, stepSize):
        for width in range(0, imWidth, stepSize):
            yield(width, height, image[height: height+boxSize[1], width:width+boxSize[0]])

def predictAverage(im, model):
    print('Predicting Average of Image...')
    imHeight, imWidth = im.shape[0], im.shape[1]
            
    boxWidth = 150
    boxHeight = 150
    total_box = 0
    num_box = 0
    scalar = 1
    for scaleImage in imagePyramid(im, scale=1.25):
        scalar = 1.25 * scalar
        for(x,y,box) in boundingBox(scaleImage, imHeight, imWidth, stepSize=32, boxSize=(boxWidth, boxHeight)):
            if box.shape[0] != boxHeight or box.shape[1] != boxWidth:
                continue
        
            clone = scaleImage.copy()
            box = clone[y: y+boxWidth, x:x+boxHeight]
            showBox = cv2.resize(box, (50, 50))
            reshapeBox =  showBox
            reshapeBox = np.reshape(reshapeBox, [-1, 50, 50, 3])
            reshapeBox = reshapeBox/255.0
        
            predict = model.predict(reshapeBox)
            if(predict[0] < .96):                        
                total_box = total_box + y * scalar
                num_box += 1
                cv2.rectangle(im, (x,y), (x + boxWidth, y + boxHeight), (0, 0, 255), 2)
                cv2.imwrite('predicted2.jpg', im)
   
        
        
        cv2.imwrite('predictedStage3.jpg', im)
        



    return(int(total_box/num_box))
'''
model = load_model('model.h5')

im = im/255.0

'''