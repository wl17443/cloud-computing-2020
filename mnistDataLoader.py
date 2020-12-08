import numpy as np 
from keras.datasets import mnist  

def load_data():
  (train_x, train_y), (test_x, test_y) = mnist.load_data()

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
    train_y_formatted[x, train_y[x]] = 1.0
  for x in range(len(test_y)):
    test_y_formatted[x, test_y[x]] = 1.0 
  return (train_x_formatted, train_y_formatted), (test_x_formatted, test_y_formatted)
