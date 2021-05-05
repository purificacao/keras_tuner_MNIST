from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras import optimizers

from kerastuner import HyperModel
    
class SearchModel(HyperModel):
    
    '''hp is an object which is internally passed by tuner to model-building function to help it specify range of hyperparameter values whenever the tuner is called.
    
    it's possible to use hp.Float, hp.Choice and hp.Int. 
    
    For exemple, to tune the number of filters in Convolutional Neural Networks or number of units in a Dense layer, we can use hp.Int('units', min_value = 32, max_value = 128, step = 32);
    
    To tune float hyperparameters, such as dropout rate or learning rate, we use:
    
    hp.Float('dropout', 0, 0.5, step=0.1)
    
    To tune functions to be used (activation functions) or specific values within a range, we can use:    
    
    hp.Choice("dense_activation", values=["relu", "tanh", "sigmoid"]'''
    
       
    def __init__(self, shape, num_classes):
        self.shape = shape
        self.num_classes = num_classes   
    
    def build(self, hp):
        model = Sequential()
        model.add(Conv2D(filters = 28, kernel_size=(5,5), activation='relu', padding='same', input_shape = self.shape))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        
        model.add(Dropout(
                rate=hp.Float(
                    "dropout_0", min_value=0.0, max_value=0.5, default=0.25, step=0.25,
                    )
              )
        )       
                
        model.add(Conv2D(14, kernel_size=(5,5),  activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Flatten())
     
        model.add(Dense(128, activation='relu'))
        
        model.add(Dropout(
                rate=hp.Float(
                    "dropout_1", min_value=0.0, max_value=0.5, default=0.25, step=0.25,
                    )
              )
        )               
           
        model.add(Dense(self.num_classes, activation = "softmax"))   
        
        model.compile(
            optimizer=optimizers.SGD(
                    lr=1e-3,                
                    decay = hp.Choice("decay", values = [1e-7, 1e-6, 1e-5]),
                    momentum = hp.Choice("momentum", values = [0.0, 0.7, 0.9, 0.95]),
                    nesterov = True),        
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
          )
                   
        return model    