## Notes:

[1] Observed improvement in ozaboost accuracy when the base classifiers are stopped training as soon as the error is slightly better than 0.5

Result:
```
Weak Learner 0  Training Accuracy:  0.626373626374
Weak Learner 0  Testing Accuracy:  0.631578947368
Weak Learner 1  Training Accuracy:  0.626373626374
Weak Learner 1  Testing Accuracy:  0.631578947368
Weak Learner 2  Training Accuracy:  0.626373626374
Weak Learner 2  Testing Accuracy:  0.631578947368
Weak Learner 3  Training Accuracy:  0.626373626374
Weak Learner 3  Testing Accuracy:  0.631578947368
Weak Learner 4  Training Accuracy:  0.863736263736
Weak Learner 4  Testing Accuracy:  0.798245614035
Weak Learner 5  Training Accuracy:  0.626373626374
Weak Learner 5  Testing Accuracy:  0.631578947368
Weak Learner 6  Training Accuracy:  0.626373626374
Weak Learner 6  Testing Accuracy:  0.631578947368
Weak Learner 7  Training Accuracy:  0.556043956044
Weak Learner 7  Testing Accuracy:  0.517543859649
Weak Learner 8  Training Accuracy:  0.626373626374
Weak Learner 8  Testing Accuracy:  0.631578947368
Weak Learner 9  Training Accuracy:  0.626373626374
Weak Learner 9  Testing Accuracy:  0.631578947368
Running for 10 weak learners..
Shape of Train Data:  (455, 30) (455,)
Shape of Test Data:  (114, 30) (114,)
Accuracy on Training Set:
0.905115076085
Test Accuracy
0.956140350877
```

As compared to earlier:
```
Weak Learner 0  Training Accuracy:  0.83956043956
Weak Learner 0  Testing Accuracy:  0.80701754386
Weak Learner 1  Training Accuracy:  0.681318681319
Weak Learner 1  Testing Accuracy:  0.640350877193
Weak Learner 2  Training Accuracy:  0.87032967033
Weak Learner 2  Testing Accuracy:  0.850877192982
Weak Learner 3  Training Accuracy:  0.804395604396
Weak Learner 3  Testing Accuracy:  0.745614035088
Weak Learner 4  Training Accuracy:  0.923076923077
Weak Learner 4  Testing Accuracy:  0.894736842105
Weak Learner 5  Training Accuracy:  0.885714285714
Weak Learner 5  Testing Accuracy:  0.80701754386
Weak Learner 6  Training Accuracy:  0.608791208791
Weak Learner 6  Testing Accuracy:  0.526315789474
Weak Learner 7  Training Accuracy:  0.661538461538
Weak Learner 7  Testing Accuracy:  0.675438596491
Weak Learner 8  Training Accuracy:  0.907692307692
Weak Learner 8  Testing Accuracy:  0.850877192982
Weak Learner 9  Training Accuracy:  0.925274725275
Weak Learner 9  Testing Accuracy:  0.894736842105
Running for 10 weak learners..
Shape of Train Data:  (455, 30) (455,)
Shape of Test Data:  (114, 30) (114,)
Accuracy on Training Set:
0.887038887248
Test Accuracy
0.90350877193
```

## To Do:

[//]: # (1. Experiment with Priming - By running it in batch mode with some initial subset of Training data and running in online mode %for the rest of Training set. Primed OzaBoost with Decision Trees performs comparably with AdaBoost.)

[//]: # (2. Compare the base models' error rates under batch and online boosting.3. Compare what the Adaboost and Ozaboost models learn and the weights assigned finally.Check the decision tree rules learnt and weights assigned)
