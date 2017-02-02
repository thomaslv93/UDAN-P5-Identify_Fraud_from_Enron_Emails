## P5 Identity Fraud From Enron Email

### Questions
__1. Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]__

The goal of this project is to use financial and email data from the members of the Enron corporation to determine which people to further investigate in regards to their potential participation in the criminal accounting fraud scandal that broke in 2002. Machine learning will allow us to learn from the data and be able to predict based on the characteristics of new data whether the person in question was indeed a "Person of Interest" (POI) worthy of further investigation. The dataset consists of 146 data points representing 146 people. There are 18 persons of interest among them leaving 128 people who are of no interest. Each data point has 21 features. All features have some entries which are missing except for the feature 'poi'. Below these features and the number of missing entries are listed:

| feature | NaN |
| :----- | :-----: |
| poi | 0 |
| total_stock_value | 20 |
| total_payments | 21 |
| email_address | 35 |
| restricted_stock | 36|
| exercised_stock_options | 44 |
| salary | 51 |
| expenses | 51 |
| other | 53 |
| to_messages | 60 |
| from_messages | 60 |
| from_poi_to_this_person | 60 |
| from_this_person_to_poi | 60 |
| bonus | 64 |
| long_term_incentive | 80 |
| deferred_income | 97 |
| deferral_payments | 107 |
| restricted_stock_deferred | 128 |
| director_fees | 129 |
| loan_advances | 142 |

There is one notable outlier in the dataset which is particularly visible when we plot the datapoints on a graph of salary and bonus. This outlier has the name TOTAL and is the aggregate of all the financial data which appears to have been mistakenly included in the dataset. I removed this datapoint before proceeding with the project.

__2. What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “properly scale features”, “intelligently select feature”]__

Because the two algorithms I will be testing (Naive Bayes and Decision Trees) do not require feature scaling since they do not make calculations based on Euclidean distance I did not use feature scaling. I started by manually selecting relevant features using a combination my human intuition and eliminating features who were missing more than half of their entries. This left 'poi', 'salary', 'to_messages', 'total_payments', 'bonus', 'total_stock_value', 'shared_receipt_with_poi', 'exercised_stock_options', 'from_messages', 'other', 'from_poi_to_this_person', 'from_this_person_to_poi', 'expenses', and 'restricted_stock'. I then created my own feature which captured the percentage of emails a person was involved in which were related to a POI. 'poi_email_density' is an aggregate feature composed of the sum of from_poi_to_this_person and from_this_person_to_poi divided by the sum of to_messages and from_messages. This feature gives us the proportion of emails sent or received by this person which had a POI on either end. I suspected this might be a more balanced way of judging a given person's connection with POIs since some people may send more emails than they receive or may send many emails to POIs but only in the context that they send many emails in general and are not themselves POIs. I felt that this feature might better capture just how much any given person was communicating with POIs even if they do not generally communicate very much, even if they do not send or receive many emails. I then used SelectKBest to select the 4 best features. The result follows:

| feature | score |
| :----- | :-----: | 
| 'exercised_stock_options' | 25.097541528735491 |
| 'total_stock_value' | 24.467654047526398 |
| 'bonus' | 21.060001707536571 | 
| 'salary'| 18.575703268041785 |
| 'restricted_stock' | 9.3467007910514877 |
| 'total_payments' | 8.8667215371077717 |
| 'shared_receipt_with_poi' | 8.7464855321290802 |
| 'expenses' | 6.2342011405067401 | 
| 'poi_email_density' | 5.5185055438125579 |
| 'from_poi_to_this_person' | 5.3449415231473374 |
| 'from_this_person_to_poi' | 2.4265081272428781 |
| 'to_messages' | 1.6988243485808501 |
| 'from_messages' | 0.16416449823428736 |

There is a very clear, sharp drop off in terms of score after the first four features which justifies our focus on the top four features (from roughly 18 to roughly 9).

It seems that my created feature did not make the cut, however interestingly enough it seems to have been more relevant than the features it combined.

| feature | score |
| :----- | :-----: |
| 'poi_email_density' | 5.5185055438125579 |
| 'from_poi_to_this_person' | 5.3449415231473374 |
| 'from_this_person_to_poi' | 2.4265081272428781 |
| 'to_messages' | 1.6988243485808501 |
| 'from_messages' | 0.16416449823428736 |

__3. What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]__

I tested both Gaussian Naive Bayes and a Decision Tree algorithm. Both of these algorithms work well for supervised discrete classification problems and are very reliable. The Decision Tree algorithm will offer the opportunity of having its parameters tuned. In the end I wound up using the Gaussian Naive Bayes classifier since its performance was better. Below are the results from both of the algorithms.

Gaussian Naive Bayes:

```
Precision: 0.48327
Recall: 0.32500
```

Decision Tree:

```
Precision: 0.42514
Recall: 0.15050
```

__4. What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric item: “tune the algorithm”]__

Parameter tuning involves the changing, fine tuning and adjustment of parameters and the testing of those parameters in order to achieve a better score on your preferred evaluation metric. We have to be careful when parameter tuning not to effectively overfit or create a classifier which does not generalize well to other data because the parameters are tuned for a very specific scenario. I used parameter tuning for the Decision Tree algorithm using GridSearchCV. The resulting parameters follow:

```
{'splitter': 'random', 'max_leaf_nodes': 8, 'min_samples_leaf': 1, 'criterion': 'gini', 'min_samples_split': 4, 'max_depth': None}
```

__5. What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric item: “validation strategy”]__

Validation is the testing of your classifier on a test set of data. Validation is crucial to prevent overfitting. A classic mistake of validation is testing on the same data you trained on. This is almost guaranteed to result in overfitting and poor generalization. Using the train_test_split function in the sklearn.cross_validation library, I split the data into a training set and a testing set and fit the data to the training set, then tested it against the test set. This resulted in evaluation metric scores which left some to be desired:

```
Accuracy Score:  0.8
Precision Score:  0.2
Recall Score:  0.2
```

However using the StratifiedShuffleSplit method in the test_classifier method of tester.py was far more effective.

__6. Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]__

Accuracy, Precision and Recall are the three evaluation metrics I used. Accuracy consists of the percentage of accurate predictions out of total predictions. Precision is the amount of true positives out of the amount of total positives, so how many times did we identify a POI who was actually a POI? Recall is the amount of true positives out of the amount of true positives and false negatives, in other words what percentage of actual POIs did we correctly identify?

Our final performance on these metrics using the test_classifier method with the StratifiedShuffleSplit follows:

```
Precision: 0.48327
Recall: 0.32500
```

This means that if we identify 100 POIs, probably roughly 48 of them will actually be interesting. And if there are 100 POIs, we will accurately identify 32 of them.