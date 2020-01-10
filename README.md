# Stage3DeepLearning

predictFrame.py:
For stage 3 I create a function that outputs the average Y value for any corn-stalk connections in a given image. 

Stage3.py:
I extract 4-5 frames from each video at set intervals and save them in a folder. Then I traverse that folder and use my prediction code to detect any connections in the model. Then I take the average value from all images and average those out as well. I store them in a list. After each video I delete the images. After the training data(4' video and 2' video values are stored in different lists) is processed, I repeat the same method onto the training data. Then each value is stored in a csv file.

predictStage3.py:
I load up all the csv files and concatenate the training data as well as the height values that are given. I manually created a csv since the data was small enough to not need code to do it for me. Then I fitted a linear regression model using scikit-learn. Then I took the data gathered from the test videos and predicted the values using my regression model.
