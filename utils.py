#File to load and prepare the data for the CNN

from tensorflow.keras.datasets import mnist

def load_data():
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    # Normalizing the data between 0 and 1, dividing by the max RGB value;i.e; 255
    x_train /= 255
    x_test /= 255
     
    return x_test, x_train, y_test, y_train


    