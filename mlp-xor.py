import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report as cr
from sklearn.metrics import confusion_matrix as cm 
def sigmoid(x):
 return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
 return x * (1 - x)
inputs = np.array([[0, 0],
 [0, 1],
 [1, 0],
 [1, 1]])
outputs = np.array([[0], [1], [1], [0]])
def forward_pass(inputs, hidden_weights, hidden_bias, output_weights, output_bias):
 hidden_layer_activation = np.dot(inputs, hidden_weights)
 hidden_layer_activation += hidden_bias
 hidden_layer_output = sigmoid(hidden_layer_activation)
 output_layer_activation = np.dot(hidden_layer_output, output_weights)
 output_layer_activation += output_bias
 predicted_output = sigmoid(output_layer_activation)
 return predicted_output, hidden_layer_output
def backward_pass(expected_output, predicted_output, output_weights, hidden_layer_output):
 error = expected_output - predicted_output
 d_predicted_output = error * sigmoid_derivative(predicted_output)
 error_hidden_layer = d_predicted_output.dot(output_weights.T)
 d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
 return d_predicted_output, d_hidden_layer
lr = 0.1
epochs = 15000
inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2, 3, 1
def train(epochs, lr, inputs, inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons):
 hidden_weights = np.random.uniform(size=(inputLayerNeurons, hiddenLayerNeurons))
 hidden_bias =np.random.uniform(size=(1, hiddenLayerNeurons))
 output_weights = np.random.uniform(size=(hiddenLayerNeurons, outputLayerNeurons))
 output_bias = np.random.uniform(size=(1, outputLayerNeurons))
 for epoch in range(epochs):
 predicted_output, hidden_layer_output = forward_pass(inputs, hidden_weights, hidden_bias, output_weights, 
output_bias)
 d_predicted_output, d_hidden_layer = backward_pass(outputs, predicted_output, output_weights, 
hidden_layer_output)
 output_weights += hidden_layer_output.T.dot(d_predicted_output) * lr
 output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * lr
 hidden_weights += inputs.T.dot(d_hidden_layer) * lr
 hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * lr
 
 return predicted_output
predicted_output = train(epochs, lr, inputs, inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons)
xor_pred = []
print('\n Rounded value of the predicted labels:\n')
for val in predicted_output:
 print(np.round(val)[0])
 xor_pred.append(np.round(val)[0])
 
print("XOR gate perceptron confusion matrix: ")
print(cm(outputs, xor_pred))
print("XOR gate perceptron metrics: ")
print(cr(outputs, xor_pred))
