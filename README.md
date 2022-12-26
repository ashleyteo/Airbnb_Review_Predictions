# Airbnb_Review_Predictions
Predicting Airbnb's property ratings in NYC

## Table of contents
* [Business_Understanding](#Business_Understanding)
* [Data_Understanding](#Data_Understanding)
* [Data_Preparation](#Data_Preparation)
* [Modeling](#Modeling)
* [Evaluation](#Evaluation)
* [Deployment](#Deployment)
* [References](#References)

## Business_Understanding
After deciding to become an Airbnb host, many prospective hosts will wonder “What makes a successful Airbnb property?” Many of the hosts think that competitive pricing is the most important factor in becoming successful in the Airbnb business. However, we believe that property ratings have a higher correlation to an Airbnb property’s success. Based on the heatmap for review ratings (refer to appendix), we can see that the review ratings are randomly scattered. Therefore, we plan to develop a model to guide business owners to strategically allocate resources to improve their listings’ review ratings. Previous customers’ high satisfaction (high ratings) can lead to more customers booking the corresponding property ( higher occupancy rates), which can lead to higher revenue. Therefore, by utilizing different variables, (such as: neighborhood, price, cancellation policy, number of reviews, etc.) we will run different supervised models to predict the ratings (Scale of 1 to 5) of various Airbnb properties. Our data mining solution will address the business problem by bringing tremendous value to prospective Airbnb hosts, as our model can specify the different factors that lead to a highly rated property.

## Data_Understanding
Before becoming an Airbnb host, the most common question they will likely have is how attractive their Airbnb list will be. Therefore, being able to predict the popularity of their listing will be important for those who want to become Airbnb hosts. In our case, we are using an Airbnb dataset consisting of 26 variables to predict the rating of Airbnb properties in New York City. The dataset includes listings, full descriptions, average review score, reviews, unique id for each reviewer, detailed comments, calendar, listing id, price and availability for that day. Since there is no direct measurement of the popularity, our team decided to use the review rating number as a proxy for popularity. The data volume is about 1.03 million observations, obtained from this data source.

## Data_Preparation
Issues that we’ve faced: Initially, we had trouble deciding how to deal with null and missing values. We were debating if we should either delete all NULL and missing values or delete NULL and missing values for numerical variables and assign NULL and missing values as a certain name for categorical variables. However, we decided to remove all NULL and missing values as they were few relative to the number of observations we have in our dataset.
Variables to delete (8): Id, Name, Hostid, hostname, License, Country, Country.code, House Rules.
We decided to get rid of these 8 columns in the original data set. The "id", "name", "host.id", "host.name", "license" columns are too trivial and have very little or no relation with the variable "review.rate.number" that we are interested in from an empirical perspective. "country" is just the US. The other column "house_rules" may be one of the factors that can impact review rate, but due to the difficulty in processing natural language and the time we have, we decide not to include this column in our further analysis.
Variables to filter (18):

- Host_identity_verified, Neighborhood, Number of reviews, Reviews per month, Review rate number: we deleted all the null and missing values since these may be miscollected data and including them may potentially mislead our conclusion.

- Neighborhood group: we deleted the null values and corrected misspelled neighborhood groups since these noisy data may potentially mislead our conclusion.

- Lat, Long: As numeric value, replacing the null value with mean value may be considered as manipulating the data. So we chose to delete the null value for these two variables since the number of null values are small compared to our dataset size.

- Instant_bookable: We identified 105 rows NAs for this column. Since this is a small proportion of the dataset, we decided to remove these rows.

- Cancellation_policy, Room type: There is no cleaning required since these are categorical types with no NAs.

- Construction year: We identified about 200 rows with NAs for this column. Since this is a small proportion of the dataset, we decided to remove these rows. We checked that the minimum year is 2003 and maximum year is 2022 which is as expected.

- Price: We re-formatted the column as a numeric with 2 decimals to remove ‘$’, and deleted the rows having blanks and NAs.

 - Service fee: We re-formatted the column as a numeric with 2 decimals to remove ‘$’. Also, we deleted the rows having blanks and NAs.
 
- Minimum nights: we deleted the rows with blanks, NAs, negative values. Also, we deleted the rows where min_nights were greater than 365 days as there are only 35 rows having data>365, which is a small proportion of the entire dataset.

- Last review: Firstly, we deleted all the null and missing values. Then deleted all the values after October 1st 2022, as having dates from the future does not make sense. Also, dates before 2015, as older listing prices will be less relevant.

- Calculated host listings count: First, we deleted all the null and missing values from the dataset. Then, we noticed that the mean value was above 7, when the median was 1. This indicates that there are a lot of outliers. Therefore, we deleted all the Outliers above 5. (Outliers = Q3+IQR)

- Availability 365: First, we deleted all the null and missing values from the dataset. Then, we deleted all the negative values, as the listing having negative days available (in a year) is unfeasible. Also, we deleted all the values above 365, as the listing being available for more than 365 within a year is impossible.

## Modeling
We have attempted to model our data with the following methods: PCA, elastic net, null, linear, lasso, random forest, KNN. We first applied the null model to set a baseline for other models. For principal component analysis, we used to reduce the amount of variables until it only contained the most important features. The pros of PCA is that it can help us to find latent features. However, using the PCA may cause the independent variable to become less interpretable and lead to a loss in information. We also used an elastic net since it helps to deal with collinearity by combining ridge and lasso. However, it requires more computing power especially if we want to apply it in k-fold cross-validation. Linear Models are beneficial in that you can identify how one marginal increase of any variable affects the review rating. However, not every data point may fit in the linear regression mode; thus, accounting for high variance in errors. The Lasso model is useful when there are lots of variables in the data and there is a need to get rid of the useless variables (by increasing the value of lambda); however, the lasso model may not be the most accurate model at all times. For example, Ridge regression models are more accurate when all variables are relevant. Random Forest models are beneficial in that it improves the accuracy of the model by randomly running the tree classification multiple times. However, a drawback of the random forest model is that we will not know the beta; therefore, we will never know the marginal effect various variables have on the dependent variable. One last model we tested is KNN. By testing the KNN model, we are able to form non-linear boundaries when grouping the datasets. This is beneficial as grouping the datasets using a KNN model is very flexible. Furthermore, KNN has an advantage in that they can be flexible when new data points come along. However, if the data has omitted variables, the data could be grouped mistakenly and create misleading interpretations.
If time permitted, one of the alternative models we would try is a neural network since it has more sophisticated predictability and theoretically more precise prediction power.

We have found the random forest to be the best applicable model for our case (based on MSE and 𝑅2 values from k-fold cross-validation). The low out-of-sample 𝑅2 values and high mean squared errors for all models except the random forest model makes us hypothesize that the relationship between the different variables and the rating is not linear. In conclusion, although the random forest does not give us the beta (so we can’t find the marginal effect of different variables on rating), the random forest model is very beneficial in informing us which variables are important in determining the rating of the Airbnb property.

## Evaluation
For the models: KNN, RandomForest, Lasso, SimpleLinear, and Null we split our data into 90% for training and 10% for testing with 10-fold cross-validation. Measures of the performance of predictive models were done through out-of-sample MSE (mean-squared error) and 𝑅2 values. Below is the graph where we can observe that the 𝑅2 value for RandomForest is around 0.4 which is better than the other models. Also, we see that the KNN model has a negative 𝑅2 which indicates a worse performance than the null model.

## References
Azmoudeh, Arian. “Airbnb Open Data.” Kaggle, 1 Aug. 2022, https://www.kaggle.com/datasets/arianazmoudeh/airbnbopendata?sort=most-comments.
