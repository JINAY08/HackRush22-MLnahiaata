Our team decided to use the Random Forest Regression approach for the ML challenge problem. First, we built the basic model and applied our model to test data, and found some results. Of course, it was not quite good, but each of our team members is completely new to Machine Learning, so it was a good start for us. 

In our model, we used train data to train the model and then test it for test data. We split the data in the size of 0.8 at random state 1. After applying the model to test data successfully, we needed to improve our model by increasing the accuracy. So, we used the Hyperparameter tuning approach. Firstly, we have identified all the parameters and their current values and then observed which is more useful for improving accuracy. Then, we have defined the acceptable range for all parameters accordingly. We decided on the number of estimators in 35 sections(changeable for different iterations). 

We have used GridSearchCV for Hyperparameter tuning. We did the tuning many times by taking the different values of parameters and their number of folds. We have used ten folds for several candidates to get the values of the most appropriate parameters on which we can apply our model so it could give the maximum accuracy. After completing this tuning, we apply the most efficient values of parameters to test data and then find the accuracy. Finally, we got 85% accuracy. It is not very good, but it is quite good compared to what we got earlier and what we got now. So, this is how we completed the challenge.
