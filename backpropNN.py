import os
import csv 
import argparse
import datetime 
import numpy as np 
import matplotlib.pyplot as plt 
from os.path import join 
from mnistDataLoader import MnistDataloader


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
      self.Weights[i] -= self.LearningRate*self.ChangeInWeights[i]

  # def calculate_accuracy(self, targets):
  #   success = 0 
  #   for i in range(len(self.Predictions)):
  #     pred = np.argmax(self.Predictions[i])
  #     if pred == np.argmax(targets[i]):
  #       success += 1
  #   return success/len(self.Predictions)

  def train_one_batch(self, batch_number):

    if batch_number % 30 == 0:
      self.LearningRate /= 2

    inputs, targets = generate_batch(train_x, train_y, self.BatchSize)
    self.forward(inputs)
    self.backprop(inputs, targets)
    self.update_weights()
    loss = (1/(self.BatchSize*10))*np.sum((self.Predictions-targets)**2)
    return loss
  
  def train(self, nr_batches):
    losses = []
    for i in range(nr_batches):
      loss = self.train_one_batch(i)
      losses.append(loss)
    return losses 

  def test(self):
    successes = 0 
    test_len = len(test_x)
    for _ in range(int(test_len/self.BatchSize)):
      inputs, targets = generate_batch(test_x, test_y, self.BatchSize) 
      self.forward(inputs)
      for j in range(self.BatchSize):
        prediction = np.argmax(self.Predictions[j])
        target = np.argmax(targets[j])
        if prediction == target:
          successes += 1
    success_rate = successes/test_len
    return success_rate

def generate_batch(dataset_x, dataset_y, batch_size):
  rand_inds = np.random.randint(0, len(dataset_x), batch_size)
  inputs = dataset_x[rand_inds]
  outputs = dataset_y[rand_inds]
  return inputs, outputs 

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Training Parameters')
  parser.add_argument('--batch_size', required =True, type=int, help='batch size')
  parser.add_argument('--nr_batches', required =True, type=int, help='number of batches')
  parser.add_argument('--learning_rate', required =True, type=float, help='learning rate')
  parser.add_argument('--hidden_layers', required =True, nargs='*', type=int, help='hidden layer sizes')

  print("Importing dataset from volume")
  input_path = '/training/Datasets'
  training_images_filepath = join(input_path, 'train-images-idx3-ubyte')
  training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte')
  test_images_filepath = join(input_path, 't10k-images-idx3-ubyte')
  test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte')

  mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
  (train_x, train_y), (test_x, test_y) = mnist_dataloader.load_data()

  args = parser.parse_args()
  architecture = [784]
  for i in range(len(args.hidden_layers)):
    architecture.append(args.hidden_layers[i])
  architecture.append(10)
  
  network = NeuralNetwork(architecture, int(args.batch_size), float(args.learning_rate))
  
  log_file = open("/training/Logs/log_file.txt", "a")

  try:
    loss = network.train(int(args.nr_batches))
  except:
    log_file.write('{}-{}-{}-{}: ERROR: Training unsuccessful.\n').format(args.batch_size, args.nr_batches, args.learning_rate, *args.hidden_layers)
    exit()

  success_rate = network.test()

  # Write to EBS checkpoint data bank 
  with open('/training/Results/training-{}-{}-{}-{}.csv'.format(args.batch_size, args.nr_batches, args.learning_rate, *args.hidden_layers), mode='w') as data:
    data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    data_writer.writerow([success_rate])
    data_writer.writerow(loss)
    data_writer.writerows(network.Weights)

  log_file.write('{}-{}-{}-{}: SUCCESS: Training complete.\n').format(args.batch_size, args.nr_batches, args.learning_rate, *args.hidden_layers)
  log_file.close()