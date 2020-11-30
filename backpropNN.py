import numpy as np 
import os
from mnistDataLoader import MnistDataloader
import matplotlib.pyplot as plt 

# Define Neural Network class 
class NeuralNetwork():
  def __init__(self, structure, batch_size, learning_rate):
    self.Structure = structure
    self.BatchSize = batch_size
    self.LearningRate = learning_rate
    self.Weights = self.init_weights(structure)
    self.Layers  = self.init_layers(structure)
    self.ActivatedLayers = self.init_activated_layers(structure)
    self.Predictions = np.zeros((batch_size, structure[-1]))
    self.ChangeInWeights = self.init_weight_change(structure)

  def sigmoid(self, a):
    return 1/(1+np.exp(-a))

  def sigmoid_prime(self, a):
    return self.sigmoid(a)*(1-self.sigmoid(a))
    
  def init_weights(self, structure):
    weights = []
    for i in range(len(structure)-1):
      weights.append(np.random.randn(structure[i], structure[i+1])/np.sqrt(structure[i]))
    return weights

  def init_layers(self, structure):
    layers = []
    for i in range(1, len(structure)):
      layers.append(np.zeros((self.BatchSize, structure[i])))
    return layers

  def init_activated_layers(self, structure):
    layers = []
    for i in range(1, len(structure)-1):
      layers.append(np.zeros((self.BatchSize, structure[i])))
    return layers

  def init_weight_change(self, structure):
    weight_change = []
    for i in range(len(structure)-1):
      weight_change.append(np.zeros((structure[i], structure[i+1])))
    return weight_change

  def forward(self, inputs):
    # print("Inputs: ", inputs)
    if len(self.Layers) == len(self.Structure):
      self.Layers[0] = inputs 
    else:
      self.Layers.insert(0, inputs)

    for i in range(len(self.Weights)):
      self.Layers[i+1] = np.matmul(self.Layers[i], self.Weights[i])
      if i < len(self.ActivatedLayers):
        self.ActivatedLayers[i] = self.sigmoid(self.Layers[i+1])

    self.Predictions = self.Layers[-1]

  def backprop(self, inputs, targets):
    dL_dPred = self.Predictions-targets 
    self.Layers[-1] = dL_dPred
    inversed_idx = list(range(1, len(self.Layers)))[::-1]
    for i in inversed_idx:
      if i-2 >= 0:
        self.ChangeInWeights[i-1] = np.matmul(self.ActivatedLayers[i-2].T, self.Layers[i]) 
        self.ActivatedLayers[i-2] = np.matmul(self.Layers[i], self.Weights[i-1].T)
        self.Layers[i-1] = np.multiply(self.sigmoid_prime(self.Layers[i-1]), self.ActivatedLayers[i-2])
      else:
        self.ChangeInWeights[i-1] = np.matmul(inputs.T, self.Layers[i]) 

  def update_weights(self):
    for i in range(len(self.Weights)):
      # print("Weight change: ", self.ChangeInWeights[i])
      self.Weights[i] -= self.LearningRate*self.ChangeInWeights[i]

  def calculate_accuracy(self, targets):
    success = 0 
    for i in range(len(self.Predictions)):
      pred = np.argmax(self.Predictions[i])
      if pred == np.argmax(targets[i]):
        success += 1
    return success/len(self.Predictions)

  def train_one_batch(self, batch_number):

    if batch_number % 30 == 0:
      self.LearningRate /= 2

    inputs, targets = generate_batch(train_x, train_y, self.BatchSize)
    self.forward(inputs)
    self.backprop(inputs, targets)
    self.update_weights()
    loss = 0.001*np.sum((self.Predictions-targets)**2)
    accuracy = self.calculate_accuracy(targets)
    return accuracy, loss
  
  def train(self, nr_batches):
    accuracies = []
    losses = []
    for i in range(nr_batches):
      accuracy, loss = self.train_one_batch(i)
      accuracies.append(accuracy)
      losses.append(loss)
    return accuracies, losses 

def generate_batch(dataset_x, dataset_y, batch_size):
  rand_inds = np.random.randint(0, len(dataset_x), batch_size)
  inputs = dataset_x[rand_inds]
  outputs = dataset_y[rand_inds]
  return inputs, outputs 

if __name__ == "__main__":
  cwd = os.getcwd()
  input_path = os.path.join(cwd, 'MNIST-dataset/Datasets') 
  training_images_filepath = os.path.join(input_path, 'train-images-idx3-ubyte')
  training_labels_filepath = os.path.join(input_path, 'train-labels-idx1-ubyte')
  test_images_filepath = os.path.join(input_path, 't10k-images-idx3-ubyte')
  test_labels_filepath = os.path.join(input_path, 't10k-labels-idx1-ubyte')

  mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)

  (train_x, train_y), (test_x, test_y) = mnist_dataloader.load_data()

  network = NeuralNetwork([784, 500, 100, 10], 10, 0.01)
  accuracy, loss = network.train(5000)
  
  plt.plot(range(5000), loss)
  plt.show()