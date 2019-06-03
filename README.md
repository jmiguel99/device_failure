# device_failure
Predict the failure of devices in Dataset
# AWS Device Failure Test
by: Jose Miguel Lopez
# Overview
This document summarises the analysis and training of a RandomForest model applied to the device_failure.csv dataset and is complementary to the jupyter notebook that contains the code and algorithms used to achieve the results presented in this document.
Question
You are tasked with building a model using machine learning to predict the probability of a device failure. When building this model, be sure to minimize false positives and false negatives. The column you are trying to predict is called failure with binary value 0 for non-failure and 1 for failure.
# The Dataset
The dataset was composed 12 columns: date, device, failure, and attribute[1-9] and it has a total of 124494 rows of records from 2015-01-01 to 2015-11-02 for a total of 304 days of records. The device column has 1163 unique classes, and the target column failure has two classes ‘0’ and ‘1’.
# Implementation
For the implementation, firstly the data is clustered by dates and devices, secondly the exploratory data analysis starts by calculating  the rates for failure, survival and replacement are calculated. Next, the columns are inspectioned to assess similarities within each other. After the inspection, all the columns values are plotted for 12 randomly picked devices to assess  the differences and correlations between variables. 
After the EDA, the dataset is transformed to drop columns and rows that are not relevant for the modelling, and also created new features for the dataset. Finally, after the dataset is cleaned and enhanced, the dataset is passed through a RandomForest model, the confusion matrix is calculated and the model evaluated and compared with its baseline.
# Exploratory Data Analysis
The dataset has a total of 124494 records, and it holds 1163 classes for each device. The total amount of reported failures is 106, this tell us that only 0.09% of devices failed over time, however, the replace_rate, which is the percentage of devices removed from the logs is  97.33%, which means that only 2.67% had an entry in the last day of the recorded data. If we have a look at fig 1, it can be noticed that in the six first dates there is a reduction of more than 400 devices, which are not related with a surge in failures. 

fig 1. Number of Devices per Day
This number also is correlated with the number of records per device which can be found in the notebook. 
The attributes 1-9 where also examined as a time series. Attributes 7 and 8 have the same values. Attribute 1 has high values which looks like noise  in comparison with the other attributes, this could mean attribute1 is made of encoded status codes.  Also device, ‘S1F0P3G2' which is a failed device, has a very characteristic plot when they fail, as seen in fig 2. From the plots it could also be noted that attributes 2, 3, 4, 7 and 8  have high number of 0s in their data, which can mean a possible error or data aggregation problems. 

fig 2. Attributes Plot

Regarding failure, it could be noted that the plots for the failed device are quite different from the rest of the other ones, which can indicate that comparing the baseline with the failures could be a good strategy for failure prediction.
Data Transformation / Feature Engineering
In this section the dataset used for the modelling is created. Attribute 8 is dropped from the dataset, and attribute 9 is renamed attribute 8 to maintain consistency. Rows corresponding to the 400 devices removed from the logs in the first 6 days are dropped, as well as the devices that survive the whole dataset. 

At the same time, the dataset is enhanced with new features: mean, standard deviation, variance, and median, and this ones are subtracted from the original columns, therefore, the model will be able to assess the delta in values. Lastly, the dataset is grouped by device and date as a mean to deal with the imbalance between the classes of the target column, and reduce the gap between them.
# Random Forest
The chosen algorithm to model the data is Random Forest due to its versatility (classification and regression), few consumption of resources, and performance. Scikit-learn offers a good tool to create models and their default configuration is capable of produce really good results. However in this case, the model quickly overfitted due to the imbalance of the dataset. Fig 3 shows the distribution of the True positive rate compared with the baseline, even though it leans towards the top left of the graph, it is still positioned over the 0 axis.


fig 3. ROC curves

This assumption is corroborated with the Confusion Matrix, which shows how all of the test dataset is classified as with Good health, or True Positives. 

fig 4. Confusion Matrix


# Conclusions
In order to produce better results, further investigation and EDA needs to be done, as well as to produce accurate results, is necessary to dive deeper into the relationships of the attributes with the failure, as well as collect better data of  the device’s failures. With more data, the algorithm will be able to produce better results and avoid the overfitting of the model due to the imbalance of the dataset.
