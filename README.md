# PoC showing how to set up Keras Tuner to fine tune CNN hyperparameters: a computer vision task.

This repository compares the results given by the three available Classes in keras Tuner library, which are: 

 - **RandomSearch**
 - **Hyperband**
 - **BayesianOptimization**

For this example, I am using the MNIST dataset, which is composed of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images. The BayesianOptimization gives a slightly better result, while the RandomSearch Class returns the worst, as seen in the table below.

|Keras Tuner Class|search_time|accuracy| 
|-----|-------|------| 
|RandomSearch|00:35:23|0.99089| 
|Hyperband|01:35:09|0.9898|
|BayesianOptimization|00:18:07|0.9901|

As one can see, the `RandomSearch` class returned the best results, taking 1/3 of the time taken by the `Hyperband` class and twice as long as the time taken by the `BayesianOptimization` class. Of course, every optimizer algorithm has its on hyperparameter to be tuned as best as one can. What I mean is, there could be a model built with a set of hyperparameters of the `Hyperband` class that would beat the others two, but something to not is how fast the `BayesianOptimization` is compared to the other two, for the chosen set of hyperparameters, returning results as good. Thus, for this very case, I would totally try and fine tune other hyperparameters of the model using the `BayesianOptimization` class!

## Requirements:

 - tensorflow==2.3
 - loguru==0.4.0
 - keras-tuner==1.0.1
 - numpy==1.19.5
 - matplotlib==3.0.3

```python

```
