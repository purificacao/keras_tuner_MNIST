# PoC showing how to set up Keras Tuner for fine tune CNN hyperparameters: a computer vision task.

This repository compares the results given by the three available Classes in keras Tuner library, which are: 

 - **RandomSearch**
 - **Hyperband**
 - **BayesianOptimization**

For this example, I am using the MNIST dataset, which is composed of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images. The BayesianOptimization gives a slightly better result, while the RandomSearch Class returns the worst, as seen in the table below.

|Keras Tuner Class|search_time|accuracy| 
|-----|-------|------| 
|RandomSearch|00:34:48|0.9893| 
|Hyperband|01:29:39|0.9915|
|BayesianOptimization|00:17:53|0.9901|
