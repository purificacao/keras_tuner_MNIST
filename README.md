# PoC showing how to set up Keras Tuner for fine tune CNN hyperparameters: a computer vision task.

This repository compares the results given by the three available Classes in keras Tuner library, which are: 

 - **RandomSearch**
 - **Hyperband**
 - **BayesianOptimization**

For this example, I am using the MNIST dataset, which is composed of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images. The BayesianOptimization gives a slightly better result, while the RandomSearch Class returns the worst.

