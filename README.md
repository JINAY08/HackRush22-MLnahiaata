Team: ML nahi aata

Team members: 
1. Haikoo Ashok Khandor, 20110071
2. Jinay Dagli, 20110084
3. Kush Patel, 20110131


Our team decided to use the Random Forest Regression approach for the ML challenge problem. First, we built the basic model and applied our model to test data, and found some results. It was not quite good, but each team member was entirely new to Machine Learning, so it was a good start. 

We used training data to train the model and used test data to test it. We split the data in the size of 0.8 at random state 1. After applying the model to test data successfully, we needed to improve our model by increasing the accuracy. So, we used the Hyperparameter tuning approach. Firstly, we identified all the parameters and their current values and then observed which is more helpful in improving accuracy. Then, we defined the acceptable range for all parameters accordingly. We decided on the number of estimators in 35 sections(changeable for different iterations). 

We have used GridSearchCV for Hyperparameter tuning. We tuned the model many times by taking different values of parameters and their number of folds. We used ten folds for several candidates to obtain the values of the appropriate parameters, which, when applied to our model, gives maximum accuracy. After completing the tuning, we used the most efficient values of parameters to test data and found the accuracy. Finally, we achieved an accuracy of 85%. It was not a good measure, but it was better than what we got earlier. That is how we attempted the challenge. It was a pretty good experience for us as this was the first time we participated in a hackathon. We want to thank the HackRush'22 team for providing this wonderful experience. 
