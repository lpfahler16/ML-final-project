# NFL Play Predictor

## Abstract
We are creating ML models from NFL play data that will help us to predict whether or not teams will go for it on 4th down, and whether or not they will succeed if they do go for it.

## Introduction
### What is the problem?
In football, teams have big decisions to make when it comes to 4th down. They can either go for the first down and risk turning the ball over, punt the ball to the other team further away, or go for a field goal to try and score 3 points. There are a variety of factors that help coaches decide what to do. However, there is a lot of gray area where they are not sure what to go with. This is why we look to apply ML models to different situations and find the best times to make each decision.

### Why is the problem interesting?
This problem leads to the loss or gain of millions of dollars for teams, as this small decision can be the main factor in whether or not they win a game. Therefore, it would be very useful for NFL coaches because it could help them make these decisions backed by real data rather than their gut. Furthermore, fans could use these models and use them to gamble on games. With a better understanding of what play is happening next, fans can more accurately perform sports betting in hopes of making money.

### What is the approach you propose to tackle the problem?
Our approach is to split a very large dataset of NFL plays into training and testing data, and then use this data and perform KNN, Random Forests, and Neural Networks. We hope to find the model that works best with this data, and allows us to predict a future play.

### Why is the approach a good approach compared with other competing methods?
The method current NFL coaches use is just going by feeling and that often does not work because a dataset will utilize better information than available to a human. Other methods such as logistic regression were considered, but it would not be as useful because we want more solid predictions rather than a probability, especially as there are multiple possibilities.

### What are the key components of my approach and results?
The key components of our approach is cleaning the dataset to get only the plays we are interested in, and classifying them as one of the four outcomes. We will likely hone in on the accuracy scores of the results to see if this method of prediction would be useful. One limitation is that coaches have different styles so we are merely getting the average of all coaches. There are more aggressive coaches and more timid coaches and this data cannot account for that.

## Setup
### Dataset
The dataset is from NFLsavant.com which provides data on every NFL play from every season for the past decade. The columns are team name, yard line, game date, seconds remaining, down, yard to go, score differential, and the targets: converted fourth down, attempted fourth down, field goal, and punt.  

### Experimental Setup
We ran 3 models for this experiment. We used Random Forests and KNN with and without cross validation as well as a neural network. For each model we tested 2 different sets of classes. The first was whether a team punted, kicked a field goal, or went for it on fourth down. The second was whether a team punted, kicked a field goal, went for it and made it, or went for it and failed on fourth down. We cross validated the random forests on max features, max depth, number of estimators, and min sample split. KNN was cross validated on number of neighbors.

### Problem Setup

## Results
### Main Results
We found that we could strongly predict whether I a team would punt, kick a field goal, or go for it with about 90% accuracy, but it was much more inconsistent to predict whether or not the 4th down would be converted. The best model in predicting what will happen on 4th down was random forests with an attempt prediction accuracy of about 90%.

### Supplementary Results

## Discussion

## Conclusion

## References
