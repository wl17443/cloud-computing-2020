import numpy as np
import struct
from array import array
import os 

class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
      self.training_images_filepath = training_images_filepath
      self.training_labels_filepath = training_labels_filepath
      self.test_images_filepath = test_images_filepath
      self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
      labels = []
      with open(labels_filepath, 'rb') as file:
          magic, size = struct.unpack(">II", file.read(8))
          if magic != 2049:
              raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
          labels = array("B", file.read())        
      
      with open(images_filepath, 'rb') as file:
          magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
          if magic != 2051:
              raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
          image_data = array("B", file.read())        
      images = []
      for i in range(size):
          images.append([0] * rows * cols)
      for i in range(size):
          img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
          img = img.reshape(28, 28)
          images[i][:] = img            
      
      return images, labels
                
    def load_data(self):
      train_x, train_y = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
      test_x, test_y = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
      
      train_x = np.array(train_x).astype('float32')/255
      test_x  = np.array(test_x).astype('float32')/255

      train_x_formatted = np.zeros((len(train_x), 784))
      test_x_formatted   = np.zeros((len(test_x), 784))

      train_y_formatted  = np.zeros((len(train_y), 10))
      test_y_formatted   = np.zeros((len(test_y), 10))

      for x in range(len(train_x)):
        for i in range(28):
          for j in range(28):
            train_x_formatted[x, i*28+j] = train_x[x, i, j]

      for x in range(len(test_x)):
        for i in range(28):
          for j in range(28):
            test_x_formatted[x, i*28+j] = test_x[x, i, j]

      for x in range(len(train_y)):
        train_y_formatted[x, train_y[i]] = 1.0 

      for x in range(len(test_y)):
        test_y_formatted[x, test_y[i]] = 1.0

      return (train_x_formatted, train_y_formatted), (test_x_formatted, test_y_formatted)
