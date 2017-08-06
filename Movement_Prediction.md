# Practical Machine Learning Final Project
Pavarit Boonyasirichok  
August 3, 2017  


  

```r
library(caret)
set.seed(99)
```
  
This is the final project of Practical Machine Learning course. An analysis is done on [Weight Lifting Exercise Dataset](http://groupware.les.inf.puc-rio.br/har). The objective is to predict how well is the exercise is done, having `classe` variable as the output.
  
The data is processed, then fitted into different model, using cross validation. Finally, we pick the model with the best accuracy and apply to the testing set.
  
## Loading and preprocessing the data
The first step is to load data from the working directory and convert the data into appropreate classes. Also we clean up the NAs in the dataset.

```r
if(!file.exists("training.csv")) {
   download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv","training.csv")
}

if(!file.exists("testing.csv")) {
   download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv","testing.csv")
}

training <- read.csv("training.csv", na.strings=c("NA","#DIV/0!", ""))
testing <- read.csv("testing.csv", na.strings=c("NA","#DIV/0!", ""))
```
  
Next, we can explore the data by calling `str`. We can see that the first 6 columns are just row numbers, user and time stamp information, which should not have any influence on the model. We can also see that there are a lot of missing data.
  

```r
str(training)
```

```
## 'data.frame':	19622 obs. of  160 variables:
##  $ X                       : int  1 2 3 4 5 6 7 8 9 10 ...
##  $ user_name               : Factor w/ 6 levels "adelmo","carlitos",..: 2 2 2 2 2 2 2 2 2 2 ...
##  $ raw_timestamp_part_1    : int  1323084231 1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
##  $ raw_timestamp_part_2    : int  788290 808298 820366 120339 196328 304277 368296 440390 484323 484434 ...
##  $ cvtd_timestamp          : Factor w/ 20 levels "02/12/2011 13:32",..: 9 9 9 9 9 9 9 9 9 9 ...
##  $ new_window              : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1 1 1 1 1 ...
##  $ num_window              : int  11 11 11 12 12 12 12 12 12 12 ...
##  $ roll_belt               : num  1.41 1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 ...
##  $ pitch_belt              : num  8.07 8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 ...
##  $ yaw_belt                : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
##  $ total_accel_belt        : int  3 3 3 3 3 3 3 3 3 3 ...
##  $ kurtosis_roll_belt      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_picth_belt     : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_yaw_belt       : logi  NA NA NA NA NA NA ...
##  $ skewness_roll_belt      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_roll_belt.1    : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_yaw_belt       : logi  NA NA NA NA NA NA ...
##  $ max_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_belt          : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_belt          : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_roll_belt     : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_pitch_belt    : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_yaw_belt      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_total_accel_belt    : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_roll_belt        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_pitch_belt          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_pitch_belt       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_pitch_belt          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_yaw_belt         : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ gyros_belt_x            : num  0 0.02 0 0.02 0.02 0.02 0.02 0.02 0.02 0.03 ...
##  $ gyros_belt_y            : num  0 0 0 0 0.02 0 0 0 0 0 ...
##  $ gyros_belt_z            : num  -0.02 -0.02 -0.02 -0.03 -0.02 -0.02 -0.02 -0.02 -0.02 0 ...
##  $ accel_belt_x            : int  -21 -22 -20 -22 -21 -21 -22 -22 -20 -21 ...
##  $ accel_belt_y            : int  4 4 5 3 2 4 3 4 2 4 ...
##  $ accel_belt_z            : int  22 22 23 21 24 21 21 21 24 22 ...
##  $ magnet_belt_x           : int  -3 -7 -2 -6 -6 0 -4 -2 1 -3 ...
##  $ magnet_belt_y           : int  599 608 600 604 600 603 599 603 602 609 ...
##  $ magnet_belt_z           : int  -313 -311 -305 -310 -302 -312 -311 -313 -312 -308 ...
##  $ roll_arm                : num  -128 -128 -128 -128 -128 -128 -128 -128 -128 -128 ...
##  $ pitch_arm               : num  22.5 22.5 22.5 22.1 22.1 22 21.9 21.8 21.7 21.6 ...
##  $ yaw_arm                 : num  -161 -161 -161 -161 -161 -161 -161 -161 -161 -161 ...
##  $ total_accel_arm         : int  34 34 34 34 34 34 34 34 34 34 ...
##  $ var_accel_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_roll_arm         : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_pitch_arm        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_yaw_arm             : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_yaw_arm          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_yaw_arm             : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ gyros_arm_x             : num  0 0.02 0.02 0.02 0 0.02 0 0.02 0.02 0.02 ...
##  $ gyros_arm_y             : num  0 -0.02 -0.02 -0.03 -0.03 -0.03 -0.03 -0.02 -0.03 -0.03 ...
##  $ gyros_arm_z             : num  -0.02 -0.02 -0.02 0.02 0 0 0 0 -0.02 -0.02 ...
##  $ accel_arm_x             : int  -288 -290 -289 -289 -289 -289 -289 -289 -288 -288 ...
##  $ accel_arm_y             : int  109 110 110 111 111 111 111 111 109 110 ...
##  $ accel_arm_z             : int  -123 -125 -126 -123 -123 -122 -125 -124 -122 -124 ...
##  $ magnet_arm_x            : int  -368 -369 -368 -372 -374 -369 -373 -372 -369 -376 ...
##  $ magnet_arm_y            : int  337 337 344 344 337 342 336 338 341 334 ...
##  $ magnet_arm_z            : int  516 513 513 512 506 513 509 510 518 516 ...
##  $ kurtosis_roll_arm       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_picth_arm      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_yaw_arm        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_roll_arm       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_pitch_arm      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_yaw_arm        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_arm             : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_yaw_arm             : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_roll_arm      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_pitch_arm     : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_yaw_arm       : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ roll_dumbbell           : num  13.1 13.1 12.9 13.4 13.4 ...
##  $ pitch_dumbbell          : num  -70.5 -70.6 -70.3 -70.4 -70.4 ...
##  $ yaw_dumbbell            : num  -84.9 -84.7 -85.1 -84.9 -84.9 ...
##  $ kurtosis_roll_dumbbell  : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_picth_dumbbell : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_yaw_dumbbell   : logi  NA NA NA NA NA NA ...
##  $ skewness_roll_dumbbell  : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_pitch_dumbbell : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_yaw_dumbbell   : logi  NA NA NA NA NA NA ...
##  $ max_roll_dumbbell       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_dumbbell      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_dumbbell        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_roll_dumbbell       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_dumbbell      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_yaw_dumbbell        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_roll_dumbbell : num  NA NA NA NA NA NA NA NA NA NA ...
##   [list output truncated]
```
  
Next, we will try to remove the columns with a lot of NAs. First, check for the percentage of missing value in the data.
  

```r
missingdata <- sapply(training, function(x){sum(is.na(x))/length(x)})
missingdata
```

```
##                        X                user_name     raw_timestamp_part_1 
##                     0.00                     0.00                     0.00 
##     raw_timestamp_part_2           cvtd_timestamp               new_window 
##                     0.00                     0.00                     0.00 
##               num_window                roll_belt               pitch_belt 
##                     0.00                     0.00                     0.00 
##                 yaw_belt         total_accel_belt       kurtosis_roll_belt 
##                     0.00                     0.00                     0.98 
##      kurtosis_picth_belt        kurtosis_yaw_belt       skewness_roll_belt 
##                     0.98                     1.00                     0.98 
##     skewness_roll_belt.1        skewness_yaw_belt            max_roll_belt 
##                     0.98                     1.00                     0.98 
##           max_picth_belt             max_yaw_belt            min_roll_belt 
##                     0.98                     0.98                     0.98 
##           min_pitch_belt             min_yaw_belt      amplitude_roll_belt 
##                     0.98                     0.98                     0.98 
##     amplitude_pitch_belt       amplitude_yaw_belt     var_total_accel_belt 
##                     0.98                     0.98                     0.98 
##            avg_roll_belt         stddev_roll_belt            var_roll_belt 
##                     0.98                     0.98                     0.98 
##           avg_pitch_belt        stddev_pitch_belt           var_pitch_belt 
##                     0.98                     0.98                     0.98 
##             avg_yaw_belt          stddev_yaw_belt             var_yaw_belt 
##                     0.98                     0.98                     0.98 
##             gyros_belt_x             gyros_belt_y             gyros_belt_z 
##                     0.00                     0.00                     0.00 
##             accel_belt_x             accel_belt_y             accel_belt_z 
##                     0.00                     0.00                     0.00 
##            magnet_belt_x            magnet_belt_y            magnet_belt_z 
##                     0.00                     0.00                     0.00 
##                 roll_arm                pitch_arm                  yaw_arm 
##                     0.00                     0.00                     0.00 
##          total_accel_arm            var_accel_arm             avg_roll_arm 
##                     0.00                     0.98                     0.98 
##          stddev_roll_arm             var_roll_arm            avg_pitch_arm 
##                     0.98                     0.98                     0.98 
##         stddev_pitch_arm            var_pitch_arm              avg_yaw_arm 
##                     0.98                     0.98                     0.98 
##           stddev_yaw_arm              var_yaw_arm              gyros_arm_x 
##                     0.98                     0.98                     0.00 
##              gyros_arm_y              gyros_arm_z              accel_arm_x 
##                     0.00                     0.00                     0.00 
##              accel_arm_y              accel_arm_z             magnet_arm_x 
##                     0.00                     0.00                     0.00 
##             magnet_arm_y             magnet_arm_z        kurtosis_roll_arm 
##                     0.00                     0.00                     0.98 
##       kurtosis_picth_arm         kurtosis_yaw_arm        skewness_roll_arm 
##                     0.98                     0.98                     0.98 
##       skewness_pitch_arm         skewness_yaw_arm             max_roll_arm 
##                     0.98                     0.98                     0.98 
##            max_picth_arm              max_yaw_arm             min_roll_arm 
##                     0.98                     0.98                     0.98 
##            min_pitch_arm              min_yaw_arm       amplitude_roll_arm 
##                     0.98                     0.98                     0.98 
##      amplitude_pitch_arm        amplitude_yaw_arm            roll_dumbbell 
##                     0.98                     0.98                     0.00 
##           pitch_dumbbell             yaw_dumbbell   kurtosis_roll_dumbbell 
##                     0.00                     0.00                     0.98 
##  kurtosis_picth_dumbbell    kurtosis_yaw_dumbbell   skewness_roll_dumbbell 
##                     0.98                     1.00                     0.98 
##  skewness_pitch_dumbbell    skewness_yaw_dumbbell        max_roll_dumbbell 
##                     0.98                     1.00                     0.98 
##       max_picth_dumbbell         max_yaw_dumbbell        min_roll_dumbbell 
##                     0.98                     0.98                     0.98 
##       min_pitch_dumbbell         min_yaw_dumbbell  amplitude_roll_dumbbell 
##                     0.98                     0.98                     0.98 
## amplitude_pitch_dumbbell   amplitude_yaw_dumbbell     total_accel_dumbbell 
##                     0.98                     0.98                     0.00 
##       var_accel_dumbbell        avg_roll_dumbbell     stddev_roll_dumbbell 
##                     0.98                     0.98                     0.98 
##        var_roll_dumbbell       avg_pitch_dumbbell    stddev_pitch_dumbbell 
##                     0.98                     0.98                     0.98 
##       var_pitch_dumbbell         avg_yaw_dumbbell      stddev_yaw_dumbbell 
##                     0.98                     0.98                     0.98 
##         var_yaw_dumbbell         gyros_dumbbell_x         gyros_dumbbell_y 
##                     0.98                     0.00                     0.00 
##         gyros_dumbbell_z         accel_dumbbell_x         accel_dumbbell_y 
##                     0.00                     0.00                     0.00 
##         accel_dumbbell_z        magnet_dumbbell_x        magnet_dumbbell_y 
##                     0.00                     0.00                     0.00 
##        magnet_dumbbell_z             roll_forearm            pitch_forearm 
##                     0.00                     0.00                     0.00 
##              yaw_forearm    kurtosis_roll_forearm   kurtosis_picth_forearm 
##                     0.00                     0.98                     0.98 
##     kurtosis_yaw_forearm    skewness_roll_forearm   skewness_pitch_forearm 
##                     1.00                     0.98                     0.98 
##     skewness_yaw_forearm         max_roll_forearm        max_picth_forearm 
##                     1.00                     0.98                     0.98 
##          max_yaw_forearm         min_roll_forearm        min_pitch_forearm 
##                     0.98                     0.98                     0.98 
##          min_yaw_forearm   amplitude_roll_forearm  amplitude_pitch_forearm 
##                     0.98                     0.98                     0.98 
##    amplitude_yaw_forearm      total_accel_forearm        var_accel_forearm 
##                     0.98                     0.00                     0.98 
##         avg_roll_forearm      stddev_roll_forearm         var_roll_forearm 
##                     0.98                     0.98                     0.98 
##        avg_pitch_forearm     stddev_pitch_forearm        var_pitch_forearm 
##                     0.98                     0.98                     0.98 
##          avg_yaw_forearm       stddev_yaw_forearm          var_yaw_forearm 
##                     0.98                     0.98                     0.98 
##          gyros_forearm_x          gyros_forearm_y          gyros_forearm_z 
##                     0.00                     0.00                     0.00 
##          accel_forearm_x          accel_forearm_y          accel_forearm_z 
##                     0.00                     0.00                     0.00 
##         magnet_forearm_x         magnet_forearm_y         magnet_forearm_z 
##                     0.00                     0.00                     0.00 
##                   classe 
##                     0.00
```
  
Here, we see there are several columns with large porportion of missing values, so we want to remove them. This time we remove columns with more than 80% NAs.
  

```r
training <- training[, !missingdata > 0.8]
testing <- testing[, !missingdata > 0.8]
```
  
Finally, we remove the first 6 columns which we intended to exclude from the model.
  

```r
training <- training[, -(1:6)]
testing <- testing[, -(1:6)]
```
  
##Model Selection
  
We will be build sevaral models, compare their performance, then pick the one with the best accuracy performance. The models used are classification trees, random forest, K-nearest neighbor and linear discriminant analysis.
  
All of these methods will be done using K-fold cross validation. First, we create `control` which is a train control object with 5 fold cross validation.
  

```r
control = trainControl(method="cv", number = 5)
```
  
Then we pass this into train function to create models. We also print the result and confusion matrix to show the accuracy.
  

```r
modrpart <- train(classe ~., method = "rpart", data = training, trControl = control)
modrpart
```

```
## CART 
## 
## 19622 samples
##    53 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 15698, 15696, 15699, 15697, 15698 
## Resampling results across tuning parameters:
## 
##   cp     Accuracy  Kappa
##   0.039  0.54      0.406
##   0.060  0.42      0.209
##   0.115  0.33      0.074
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.039.
```

```r
confusionMatrix(modrpart)
```

```
## Cross-Validated (5 fold) Confusion Matrix 
## 
## (entries are percentual average cell counts across resamples)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 24.8  5.5  4.3  4.5  1.7
##          B  0.7  7.3  0.8  3.1  2.4
##          C  2.7  6.5 12.3  8.4  4.8
##          D  0.0  0.0  0.0  0.0  0.0
##          E  0.2  0.0  0.0  0.4  9.5
##                             
##  Accuracy (average) : 0.5391
```
  

```r
modrf <- train(classe ~., method = "rf", data = training, trControl = control)
modrf
```

```
## Random Forest 
## 
## 19622 samples
##    53 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 15698, 15699, 15699, 15695, 15697 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa
##    2    1         1    
##   27    1         1    
##   53    1         1    
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

```r
confusionMatrix(modrf)
```

```
## Cross-Validated (5 fold) Confusion Matrix 
## 
## (entries are percentual average cell counts across resamples)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 28.4  0.0  0.0  0.0  0.0
##          B  0.0 19.3  0.0  0.0  0.0
##          C  0.0  0.0 17.4  0.1  0.0
##          D  0.0  0.0  0.0 16.3  0.0
##          E  0.0  0.0  0.0  0.0 18.3
##                             
##  Accuracy (average) : 0.9982
```
  

```r
modknn <- train(classe ~., method = "knn", data = training, trControl = control)
modknn
```

```
## k-Nearest Neighbors 
## 
## 19622 samples
##    53 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 15699, 15698, 15696, 15697, 15698 
## Resampling results across tuning parameters:
## 
##   k  Accuracy  Kappa
##   5  0.93      0.91 
##   7  0.91      0.89 
##   9  0.89      0.87 
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was k = 5.
```

```r
confusionMatrix(modknn)
```

```
## Cross-Validated (5 fold) Confusion Matrix 
## 
## (entries are percentual average cell counts across resamples)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 27.5  0.8  0.2  0.2  0.2
##          B  0.3 17.3  0.5  0.1  0.4
##          C  0.2  0.7 16.2  1.0  0.3
##          D  0.4  0.4  0.4 14.9  0.6
##          E  0.1  0.2  0.1  0.2 17.0
##                             
##  Accuracy (average) : 0.9282
```
  

```r
modlda <- train(classe ~., method = "lda", data = training, trControl = control)
modlda
```

```
## Linear Discriminant Analysis 
## 
## 19622 samples
##    53 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 15698, 15697, 15696, 15699, 15698 
## Resampling results:
## 
##   Accuracy  Kappa
##   0.71      0.64
```

```r
confusionMatrix(modlda)
```

```
## Cross-Validated (5 fold) Confusion Matrix 
## 
## (entries are percentual average cell counts across resamples)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 23.4  2.7  1.7  0.9  0.8
##          B  0.8 12.6  1.6  0.7  2.7
##          C  1.9  2.5 11.5  2.0  1.6
##          D  2.2  0.8  2.1 12.2  1.7
##          E  0.1  0.8  0.5  0.6 11.5
##                             
##  Accuracy (average) : 0.7127
```
  
According to results above, the random forest model has the best accuracy of 99.82%. Hence, we will be using random forest to predict the test set.
  
##Prediction Quiz
Finally, We apply the random forest model to the test set and get the final results.
  

```r
submission <- predict(modrf, newdata = testing)
names(submission) <- 1:20
submission
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```
