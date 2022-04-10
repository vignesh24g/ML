import os
import pandas as pd 
import numpy as np 
import random 
import matplotlib.pyplot as plt
class MultiLayerPerceptron: 
 def __init__(self, params=None):
 self.inputLayer = 4 
 self.hiddenLayer = 5 
 self.OutputLayer = 3 
 self.learningRate = 0.005 
 self.max_epochs = 765 
 self.BiasHiddenValue = -1 
 self.BiasOutputValue = -1 
 self.activation = self.activation['sigmoid']
 self.deriv = self.deriv['sigmoid']
 
 self.WEIGHT_hidden = self.starting_weights(self.hiddenLayer, self.inputLayer)
 self.WEIGHT_output = self.starting_weights(self.OutputLayer, self.hiddenLayer)
 self.BIAS_hidden = np.array([self.BiasHiddenValue for i in range(self.hiddenLayer)])
 self.BIAS_output = np.array([self.BiasOutputValue for i in range(self.OutputLayer)])
 self.classes_number = 3 
 
 pass
 
 def starting_weights(self, x, y):
 return [[2 * random.random() - 1 for i in range(x)] for j in range(y)]
 activation = {
 'sigmoid': (lambda x: 1/(1 + np.exp(-x * 1.0))),
 }
 deriv = {
 'sigmoid': (lambda x: x*(1-x)),
 }
 
 def Backpropagation_Algorithm(self, x):
 DELTA_output = []
 
 ERROR_output = self.output - self.OUTPUT_L2
 DELTA_output = ((-1)*(ERROR_output) * self.deriv(self.OUTPUT_L2)
 
 for i in range(self.hiddenLayer):
 for j in range(self.OutputLayer):
 self.WEIGHT_output[i][j] -= (self.learningRate * (DELTA_output[j] * self.OUTPUT_L1[i]))
 self.BIAS_output[j] -= (self.learningRate * DELTA_output[j])
 
 delta_hidden = np.matmul(self.WEIGHT_output, DELTA_output)* self.deriv(self.OUTPUT_L1)
 for i in range(self.OutputLayer):
 for j in range(self.hiddenLayer):
 self.WEIGHT_hidden[i][j] -= (self.learningRate * (delta_hidden[j] * x[i]))
 self.BIAS_hidden[j] -= (self.learningRate * delta_hidden[j])
 
 def predict(self, X, y):
 my_predictions = []
 
 forward = np.matmul(X,self.WEIGHT_hidden) + self.BIAS_hidden
 forward = np.matmul(forward, self.WEIGHT_output) + self.BIAS_output
 
 for i in forward:
 my_predictions.append(max(enumerate(i), key=lambda x:x[1])[0])
 
 return my_predictions
 def fit(self, X, y): 
 count_epoch = 1
 total_error = 0
 n = len(X);
 epoch_array = []
 error_array = []
 W0 = []
 W1 = []
 while(count_epoch <= self.max_epochs):
 for idx,inputs in enumerate(X): 
 self.output = np.zeros(self.classes_number)
 
 self.OUTPUT_L1 = self.activation((np.dot(inputs, self.WEIGHT_hidden) + self.BIAS_hidden.T))
 self.OUTPUT_L2 = self.activation((np.dot(self.OUTPUT_L1, self.WEIGHT_output) + self.BIAS_output.T))
 
 if(y[idx] == 0): 
 self.output = np.array([1,0,0]) 
 elif(y[idx] == 1):
 self.output = np.array([0,1,0])
 elif(y[idx] == 2):
 self.output = np.array([0,0,1])
 
 square_error = 0
 for i in range(self.OutputLayer):
 erro = (self.output[i] - self.OUTPUT_L2[i])**2
 square_error = (square_error + (0.05 * erro))
 total_error = total_error + square_error
 
 self.Backpropagation_Algorithm(inputs)
 
 total_error = (total_error / n)
 
 if((count_epoch % 50 == 0)or(count_epoch == 1)):
 print("Epoch ", count_epoch, "- Total Error: ",total_error)
 error_array.append(total_error)
 epoch_array.append(count_epoch)
 
 W0.append(self.WEIGHT_hidden)
 W1.append(self.WEIGHT_output)
 
 
 count_epoch += 1
 
 return self
 
iris_dataset = pd.read_csv('iris.csv')
iris_dataset.head()
iris_dataset.loc[iris_dataset['species']=='setosa','species']=0
iris_dataset.loc[iris_dataset['species']=='versicolor','species']=1
iris_dataset.loc[iris_dataset['species']=='virginica','species'] = 2
iris_label = np.array(iris_dataset['species'])
iris_data = np.array(iris_dataset[['sepal_length','sepal_width', 'petal_length', 'petal_width']]) 
Perceptron = MultiLayerPerceptron()
Perceptron.fit(iris_data, iris_label)
pred = Perceptron.predict(iris_data, iris_label)
print(pred)
print(iris_label)
from sklearn.metrics import classification_report as cr
from sklearn.metrics import confusion_matrix as cm 
label = []
for l in iris_label:
 label.append(l)
print("Confusion matrix: ")
print(cm(label, pred))
print("Metrics: ")
print(cr(label, pred))
