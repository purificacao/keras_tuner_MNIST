# PoC showing how to set up Keras Tuner to fine tune CNN hyperparameters: a computer vision task.

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

As one can see, the Hyperband Class returned the best results, but had a way longer searching time, whilst BayesianOptimization Class took only 18 minutes (almost) and returned the second best result. Of course, the results achieved here are data and parameter dependent, which means, each of the three classes can achieve great results. All it takes is to set up proper configurations of its parameters. For example, maybe the RandomSearch class could give better results with higher max_trials and higher executions_per_trial, and so on. Another thing to notice is that the hyperband and Bayesian classes return the same best hyperparameters combinations for the CNN, while the Random Class returned a slightly different combination.

## Requirements:

 - tensorflow==2.1
 - loguru==0.4.0
 - keras-tuner==1.0.1
 - numpy==1.16.2
 - matplotlib==3.0.3



```python

```
