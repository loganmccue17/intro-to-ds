---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---
# Basic Linear Regression

A standard linear regression analysis estimating the prices of various automobiles based on their categorical characteristics.


```{code-cell} ipython3
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
sns.set_theme(font_scale=2,palette='colorblind')
```

## Dataset Imports and Cleaning

```{code-cell} ipython3
# Import the Automobile Dataset from UCI Machine Learning Library
automobile_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
columns = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", 
           "body-style", "drive-wheels", "engine-location", "wheel-base", "length", "width", 
           "height", "curb-weight", "engine-type", "num-of-cylinders", "engine-size", "fuel-system", 
           "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"]
automobile_df = pd.read_csv(automobile_url, names = columns)
```

```{code-cell} ipython3
automobile_df.head()
```

```{code-cell} ipython3
# Here we replace '?' in our numerical columns with NaN, and then remove those columns as
# we do not want them to contribute with our data
automobile_df.replace('?', np.nan, inplace=True)
automobile_df.dropna(axis=0, inplace=True)
automobile_df.head()
```

```{code-cell} ipython3
automobile_df.shape
```

## Initial Dataset Descriptions

+++

The Automobile Dataset from the UC Irvine Machine Learning Repository contains numerical and categorical information on the features of a given automobile and the given price of each vehicle. This allows a linear regression model to be effectively performed, as it can be assumed that continuous attributes like 'highway-mpg' scale linearly to vehicle price. The target variable for this study would therefore be automobile price. The automobile attributes that will be used to linearly map price (in dollars) are all based on numerical values. This includes: <br>
- symboling (-3 - 3) measuring a risk factor used for insurance <br>
- normalized average loss due to insurance claims (unitless)
- wheel-base (inches)
- length of car (inches)
- width of car (inches)
- height of car (inches)
- weight of car (pounds)
- engine size (cubic inches)
- bore (inches)
- stroke (inches)
- compression ratio (unitless)
- horsepower (horsepower)
- peak rpm (revolutions per minute)
- city mpg (miles per gallon)
- highway mpg (miles per gallon)

+++

The task of this regression analysis is to examine if automobile price can be effectively predicted based on the continuous features given above for each sample. This analysis assumes that automobile price can be represented as a linear relationship between these numerical attributes. Regression requires a dataset to have a relationship that can be modeled on an x-y plane, so that a mathematical correlation can be identified. By having continuous attributes and a continuous target variable, this analysis aims to find a "slope" and "intecept" to proportionally model this relationship. This dataset is suitable for a regression analysis as each numerical attribute and the target variable have continuous numerical scales, which are assumed to have some sort of relationship with each other.

+++

## Fitting and Testing the Linear Regression Model

```{code-cell} ipython3
# Create the X and y datasets
    # X contains all numerical attributes
    # y contains the price
automobile_X_multi = automobile_df[["symboling", "normalized-losses", "wheel-base", "length", "width", 
           "height", "curb-weight", "engine-size", "bore", "stroke", "compression-ratio", "horsepower",
           "peak-rpm", "city-mpg", "highway-mpg"]]
automobile_X_multi = automobile_X_multi.astype(float)


automobile_y_multi = automobile_df['price']
automobile_y_multi = automobile_y_multi.astype(float)
```

```{code-cell} ipython3
# We split the red_X and red_y datasets into training and testing data
    # The default training percentage is 75%
automobile_X_train, automobile_X_test, automobile_y_train, automobile_y_test = train_test_split(automobile_X_multi, automobile_y_multi, random_state = 17)

# Here we print the shape of the training data and see that it is 75% of our original dataframe
automobile_X_train.shape, automobile_y_train.shape
```

```{code-cell} ipython3
regr = linear_model.LinearRegression()
```

```{code-cell} ipython3
# We fit our dataframe with our training data
regr.fit(automobile_X_train, automobile_y_train)
```

```{code-cell} ipython3
# We can then make predicted data based on our X_test
automobile_y_pred = regr.predict(automobile_X_test)
```

## Visualization of Each Attribute

```{code-cell} ipython3
# Loop through each predictor and plot
for i in range(automobile_X_test.shape[1]):

    # Create a new figure for each predictor to avoid overlap between plots
    plt.figure(figsize=(8, 6))
    
    # Access the i-th predictor using .iloc for DataFrame (: means all rows, i means the iterative columns)
    plt.scatter(automobile_X_test.iloc[:, i], automobile_y_test, label="Actual", color='blue')
    plt.scatter(automobile_X_test.iloc[:, i], automobile_y_pred, label="Predicted", color='red')

    # Set labels and title
    plt.xlabel(automobile_X_test.columns[i])
    plt.ylabel("Price ($)")
    plt.xticks(rotation = 90)
    plt.legend(fontsize = 10)
```

We can see that some attributes were very effective predictors of the price of each vehicle. Many of these trends demonstrate a linear relationship, that seems to be modeled well by the fitting algorithm. It is worth noting that while each graph models an individual numerical attribute against the vehicle price, the red predicted points have already taken into account ALL of these training variables. Here we just model the final predicted points (y_pred) against the initial X_test that was used for the fitting algorithm. That way the red points have the price target variable associated with each initial test feature.

+++

## R2 Score

+++

When using multiple attribute variables, it is often difficult to graphically visualize the prediction by regression without modeling each attribute separately. This is because while a pseudo-linear relationship was found, it is not linear with respect to each attribute, but instead a combination of ALL attributes. Due to this, it is easier to demonstrate how accurate our regression experiment was by using statistical metrics. The first of which we will use is the R2 metric, the standard in measuring linear regression.

```{code-cell} ipython3
r2 = r2_score(automobile_y_test, automobile_y_pred)
r2
```

This is not a horrible r2 score. Given that we are using many features, this r2 score could tell us many things. First, the amount of attributes used to fit and predict the data could contribute to a more profound linear relationship as there is more robust data that could contribute to vehicle price. However, this comes with a potential fault where some attributes with little to no contribution affect the model's prediction.

+++

## Mean Squared Error

+++

Before moving on, we will run another statistical metric, using the mean squared error to eventually find the mean absolute error. This can be compared to summary statistics of our price to see how significant this error is. It is worth nothing that this value involves an absolute value of each error, showing how much each deviated from 0 in either direction. When we look at individual residuals later, we see that there is error in both directions about equally.

```{code-cell} ipython3
mean_abs_error = np.sqrt(mean_squared_error(automobile_y_test, automobile_y_pred))
mean_abs_error
```

```{code-cell} ipython3
automobile_y_test.describe()
```

We can see that while the mean absolute error is relatively high (3079.32 dollars), this is actually significantly lower than the standard deviation of the test dataset. This means that while there is a very large spread in the price of automobiles in this dataset, the prediction algorithm was able to estimate the price within reason. While this is still a fairly large recorded error, it seems to support the r2 value of 0.709.

+++

## Analysis of Coefficients and Residuals

```{code-cell} ipython3
regr.__dict__
```

After this analysis of linear regression, we can look at the individual coeffecients and residuals for each attribute. The following is an array of each coefficient. Because we performed a multivariable analysis in which we used multiple predictive attributes to fit and train our model, we get an array of coefficients. Each coefficient represents one attribute. We can use this array to see what attributes had the greatest effect on our model (the higher magnitude of the coefficient). This means that our model depends on these more. We can see (based on the 10s magnitude of 3), that the attributes with the most impact are the automobiles **'bore'** and **'stroke'**. These are both measurements refering to the cylinders inside an engine, which surprised me. This may suggest that vehicle price is most influenced on the engine's dimensions.

```{code-cell} ipython3
regr.coef_
```

On the other hand, we can look at residuals for each individual sample in our y_test and y_pred. We can see that these individual residuals have a large spread. There are some values with large magnitude in the negative direction (ie. -7632) and the range is extended to the opposite extreme (ie. 9948). This seems to suggest that while the overall mean residual error and r2 values was fairly good (the predictions are reasonably effective on the aggregate level), **the Linear Regression algorithm is not too good at predicting individual automobile prices**.

```{code-cell} ipython3
residuals = automobile_y_test - automobile_y_pred
residuals
```

## Repeat Experiment 5 More Times

```{code-cell} ipython3
# Add previous r2 and mean_abs_error to a list
r2_scores = []
mean_abs_error_scores = []

r2_scores.append(r2)
mean_abs_error_scores.append(mean_abs_error)

for test in range(5):
    automobile_X_train, automobile_X_test, automobile_y_train, automobile_y_test = train_test_split(automobile_X_multi, automobile_y_multi)
    regr.fit(automobile_X_train, automobile_y_train)
    automobile_y_pred = regr.predict(automobile_X_test)

    r2 = r2_score(automobile_y_test, automobile_y_pred)
    mean_abs_error = np.sqrt(mean_squared_error(automobile_y_test, automobile_y_pred))

    r2_scores.append(r2)
    mean_abs_error_scores.append(mean_abs_error)
```

```{code-cell} ipython3
# Storing our Metrics into a Dataframe for Easy Examination
metric_df = pd.DataFrame({
    'r2 Values': r2_scores,
    'Mean Absolute Errors ($)': mean_abs_error_scores
})
metric_df
```

```{code-cell} ipython3
metric_df.describe()
```

While the model contains consistency in how well it is able to predict automobile prices on an aggregate level, demonstrated by relative consistency in the r2 values and the mean absolut errors, I would not trust the deployment of this model. This is because individual residuals (y_test - y_pred) demonstrate a wide range in the model's predictions. On any individual automobile, the prediction for a price dependent on all contributing attributes seems too varied to effectively use. As mentioned before, the aggregate metrics for the entire model are fairly consistent meaning that there may be some linear relationship present, but I think this model would be better examined with either a different regression model or by looking at only a few defining attributes (like 'bore' and 'stroke' as mentioned).

+++

## Final Interpretations on the Simple Linear Regression Model

+++

Therefore, while the performance of the linear regression algorithm is consistent enough to be accurate on aggregate, the variance in individual residuals is too strong to recommend the use of this model. While this task is too complicated to be done without machine learning, we can help the machine learning algorithm to obtain better potential results. Based on the coefficients that are largest in magnitude (and therefore more characteristic of this regression model), we can try to do this analysis again with only a few attributes instead of all numerical data. We could also attempt to optimize this analysis with another regression model to see if a more complex model is better suited.

+++

## Linear Regression Using Only One Feature

+++

In this next section, we will try fitting the model based on only one feature to see if our scores improve. If successful, we will be looking for r2 values closer to 1 (demonstrating near-linearity) and reduced mean absolute errors. Ideally, we would also see individual residuals that are more consistent, as it would mean that the model is good at overall automobile price prediction. In this analysis, we will use the attribute 'bore' as it has the highest coefficient magnitude. This means our original multi-variable analysis depended on the linearity of 'bore' the most. In terms of automobiles, 'bore' represents the diameter of an engine's cylinder, which is where the piston travels through. This could affect engine performance, and possibly linearly correlate to automobile price.

```{code-cell} ipython3
# Create new automobile_X and automobile_y datasets
automobile_X_single = automobile_df["bore"].values[:,np.newaxis]
automobile_X_single = automobile_X_single.astype(float)

automobile_y_single = automobile_df['price']
automobile_y_single = automobile_y_single.astype(float)

# Get training and fitting data from train_test_split
automobile_X_train, automobile_X_test, automobile_y_train, automobile_y_test = train_test_split(automobile_X_single, automobile_y_single, random_state = 6)

# Fit the model using training data
regr.fit(automobile_X_train, automobile_y_train)

# Predict the model using our test data (which is just "bore" values)
automobile_y_pred = regr.predict(automobile_X_test)
```

```{code-cell} ipython3
r2_score(automobile_y_test, automobile_y_pred)
```

```{code-cell} ipython3
np.sqrt(mean_squared_error(automobile_y_test, automobile_y_pred))
```

```{code-cell} ipython3
plt.figure(figsize=(8, 6))

# Access the i-th predictor using .iloc for DataFrame (: means all rows, i means the iterative columns)
plt.scatter(automobile_X_test, automobile_y_test, label="Actual", color='blue')
plt.scatter(automobile_X_test, automobile_y_pred, label="Predicted", color='red')

# Set labels and title
plt.xlabel("Bore (in)")
plt.ylabel("Price ($)")
plt.xticks(rotation = 90)
plt.legend(fontsize = 10)
```

We can see from this analysis that when looking at just one attribute like "bore", the results are even worse (based on our r2 and mean absolute error). This suggests that the data could be better represented by more than one feature, though not all of them (as we discovered earlier). To properly determine if there is a trend between automobile physical attributes and its price, there would need to be a further analysis performed on only a few, chatacterizing attributes. "bore" size could be one of these attributes, but as shown above, can not be used on its own to determine a linear trend.

+++

Based on the above analysis of the multivariable prediction model against true values, the attributes that I would be interested in combined for a further analysis would be **Bore, Automobile Length, Automobile Width, City MPG, and Highway MPG**, as these visually look like they have the most linear relationship between price and its numerical value.

+++

## Regression Model Optimization

+++

### LASSO

+++

To optimize the model, we will use the LASSO method. As determined using the simple Linear Regression model, using all numerical features did not result in effective model predictions at the individual level (and not great at the aggregate level either). Our residuals were varied significantly and our statistic metrics were not optimized. In this section we will attempt to solve this problem by using a more complicated model. **LASSO** helps in determining which features are less linearly correlated to our target vartiable (price), and does not train based on them (or at least trains less based on these attributes). This analysis will allow us to see which attributes it deems most important while ideally giving us a more effective regression.

```{code-cell} ipython3
# We use the same automoble_X_multi and automobile_y_multi datasets that we have separated earlier to make our training and testing data
automobile_X_train, automobile_X_test, automobile_y_train, automobile_y_test = train_test_split(automobile_X_multi, automobile_y_multi, random_state = 17)

# We then fit a score the LASSO model with the same parameters
lasso = linear_model.Lasso(alpha = 1)
lasso.fit(automobile_X_train, automobile_y_train)
lasso.score(automobile_X_test, automobile_y_test)
```

```{code-cell} ipython3
lasso.coef_
```

### Decision Tree

+++

Weirdly enough, it seems that with an alpha of 1, no coefficients are zero and they actually resemble the slopes given above from the linear regression model. Because of this, we are going to try a Decision Tree instead. This result from this LASSO model seems to suggest that these continuous variables are not linearly correlated to price (and none have a stronger correlation necessarily). Since all lasso coefficients model the linear regression, it is reasonable to assume that no attributes have a strong correlation to price.

```{code-cell} ipython3
# Fit the Decision Tree model with default parameters
tree = DecisionTreeRegressor(random_state = 19)
np.mean(cross_val_score(tree, automobile_X_multi, automobile_y_multi, cv = 5))
```

cross_val_score fit and scores using default parameters for DecisionTreeRegressor(). It splits the dataset into 5 folds (cv = 5) and calculate a mean score across all five folds. We can see that with the default decision tree parameters, our cross_val_score was very low at 0.46

```{code-cell} ipython3
# Here we decide which parameters to change
# I altered max_depth, min_samples_split, and min_samples_leaf

param_dt = {
    'criterion' : ['squared_error'],
    'max_depth': [None] + list(range(1, 15)),
    'min_samples_split': list(range(2,10,1)),
    'min_samples_leaf': [1, 5]
}

# All parameters aim to reduce complexity of the decision tree
```

```{code-cell} ipython3
# Fit a GridSearchCV with our DecisionTreeRegressor and our altered parameters
    # We use the training data previously split for the LASSO technique.
dt = DecisionTreeRegressor(random_state = 19)
dt_opt = GridSearchCV(dt, param_dt, cv = 5)
dt_opt.fit(automobile_X_train,automobile_y_train)
```

```{code-cell} ipython3
# Using our Decision Tree Model, we predict values for y given X_test
y_pred = dt_opt.predict(automobile_X_test)
```

```{code-cell} ipython3
# The best fitting model parameters
dt_opt.best_params_
```

The best fit model parameters differ from our default decision tree, with a mean cross_val_score of 0.46 given our random state of 19. While the default max_depth is None, this max_depth was raised to 10. The min_samples_leaf parameter stayed the same as default, but the min_samples_split increased from 2 to 4. This altering in parameters, with a similar cv of 5fold, gave us an optimized score of 0.77. While this score is much better than the default Decision Tree, it is not too much greater than the simple Linear Regression model. We will now examine whether the performance is great enough to warant the use of the more complicated Decision Tree model.

```{code-cell} ipython3
# This gives the most optimized score from our new parameters against our held-out test set
dt_opt.score(automobile_X_test,automobile_y_test)
```

```{code-cell} ipython3
# These are the metrics for optimized conditions
dt_opt_df = pd.DataFrame(dt_opt.cv_results_)
dt_opt_df[(dt_opt_df['param_max_depth'] == 10) &
    (dt_opt_df['param_min_samples_leaf'] == 1) &
    (dt_opt_df['param_min_samples_split'] == 4)]
```

```{code-cell} ipython3
# These are the metrics for default conditions
dt_opt_df[(dt_opt_df['param_max_depth'].isna()) &
    (dt_opt_df['param_min_samples_leaf'] == 1) &
    (dt_opt_df['param_min_samples_split'] == 2)]
```

We can see that the default mean_fit_time is much greater (about 8 times) that of the optimized conditions. This is the same when comparing mean_score_times too. This means that as far as time complexity goes, the optimized conditions are much more effective time-sensitive than the default conditions. Not only this, but the mean_test_score on average is almost 0.1 higher. On a scale of 0-1, this is a huge difference, with the optimized test_score for our Decision Tree Regression model being much greater than that of default parameters. This meaningful difference encourages the use of the optimized Decision Tree for this analysis.

```{code-cell} ipython3
# Increasing our Cross Validation Folds to 10
dt = DecisionTreeRegressor(random_state = 19)
dt_opt = GridSearchCV(dt, param_dt, cv = 10)
dt_opt.fit(automobile_X_train,automobile_y_train)
dt_opt.best_params_
```

```{code-cell} ipython3
# These are the metrics for optimized conditions
dt_opt_df = pd.DataFrame(dt_opt.cv_results_)
dt_opt_df[(dt_opt_df['param_max_depth'] == 5) &
    (dt_opt_df['param_min_samples_leaf'] == 1) &
    (dt_opt_df['param_min_samples_split'] == 9)]
```

```{code-cell} ipython3
# These are the metrics for default conditions
dt_opt_df[(dt_opt_df['param_max_depth'].isna()) &
    (dt_opt_df['param_min_samples_leaf'] == 1) &
    (dt_opt_df['param_min_samples_split'] == 2)]
```

When the cv is increased to 10, we see drastically different results. While the mean_fit_time is still much lower than that of the default parameters, we now see a lower mean_test_score and completely different optimal parameters. This suggest that there is not a clear trend on which parameters are best optimized (not a linear scale with parameterization). This means that the model may have been better at these optimal parameters by chance under 5-fold, and that has little meaning in how effective the model is at predicting the automobile price.
