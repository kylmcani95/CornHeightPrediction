from sklearn.linear_model import LinearRegression
import numpy as np

heightTrain = np.genfromtxt('heightTrain.csv')

dataTrain2 = np.genfromtxt('averages2Train.csv', delimiter=',')
dataTrain4 = np.genfromtxt('averages4Train.csv', delimiter=',')

dataTest2 = np.genfromtxt('averages2Test.csv', delimiter=',')
dataTest4 = np.genfromtxt('averages4Test.csv', delimiter=',')


dataTrain2 = dataTrain2.reshape(3,1)
dataTrain4 = dataTrain4.reshape(3,1)

dataTrain = np.concatenate((dataTrain2, dataTrain4), axis=1)
print(dataTrain)


dataTest2 = dataTest2.reshape(1,1)
dataTest4 = dataTest4.reshape(1,1)

dataTest = np.concatenate((dataTest2, dataTest4), axis=1)
print(dataTest)

regression = LinearRegression()
regression.fit(dataTrain, heightTrain)

pred = regression.predict(dataTest)

pred= np.rint(pred)

np.savetxt('predictedValues.csv', pred)