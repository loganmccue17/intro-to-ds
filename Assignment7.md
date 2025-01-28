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

# Basic Classification

An introductory classification analysis using data regarding the activity of breast cancer cells (malignant/benign)

```{code-cell} ipython3
import pandas as pd
import seaborn as sns
from sklearn import tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import sklearn.metrics as skmetrics
import matplotlib.pyplot as plt
sns.set_theme(palette='colorblind')
```

```{code-cell} ipython3
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

column_names = [
    "ID number", "Diagnosis", "Radius (mean)", "Texture (mean)", "Perimeter (mean)",
    "Area (mean)", "Smoothness (mean)", "Compactness (mean)", "Concavity (mean)",
    "Concave points (mean)", "Symmetry (mean)", "Fractal dimension (mean)",
    "Radius (SE)", "Texture (SE)", "Perimeter (SE)", "Area (SE)", "Smoothness (SE)",
    "Compactness (SE)", "Concavity (SE)", "Concave points (SE)", "Symmetry (SE)",
    "Fractal dimension (SE)", "Radius (worst)", "Texture (worst)", "Perimeter (worst)",
    "Area (worst)", "Smoothness (worst)", "Compactness (worst)", "Concavity (worst)",
    "Concave points (worst)", "Symmetry (worst)", "Fractal dimension (worst)"
]

cancer_df = pd.read_csv(url, header=None, names=column_names)
cancer_df.head()
```

## 7.2 -- Dataset and EDA

+++

This dataset includes a specific ID number for each test case and a diagnosis given (malignant or benign). This diagnosis can be used to compare each model's prediction using the Gaussian Naive Bayes method. Every column after that includes data based on features of the cancer tumor used for machine learning classification.

+++

The purpose of this classification would be to test the Gaussian Naive Bayes algorithm in its use to correctly diagnose the activity of each breast cancer tumor. Using various features of each test case, the algorithm should be able to find patterns separating the malignant and benign cancer tumors.

```{code-cell} ipython3
cancer_df[['Diagnosis', 'Radius (mean)']].groupby('Diagnosis').describe()
```

```{code-cell} ipython3
cancer_df[['Diagnosis', 'Texture (mean)']].groupby('Diagnosis').describe()
```

```{code-cell} ipython3
cancer_df[['Diagnosis', 'Perimeter (mean)']].groupby('Diagnosis').describe()
```

When doing some brief Exploratory Data Analysis looking at the Radius, Texture, and Perimeter columns, we see that the means and standard deviations for some columns are out of range. The Radius and Perimeter have standard deviations that are statistically different when comparing benign and malignant cancer tumors. However, other columns like Texture contain overlapping error bars. This likely means that some breast cancer tumors are more characteristic for a benign or malignant diagnosis than others. Due to the statistical separation of some variables, I would hypothesize that the classification algorithm will result in reasonably high accuracy, but may be lowered by the presence of less characteristic variables. Due to this, there are likely to be a significant amount of false positives or false negatives which would lower the amount of true diagnoses needed for 100% accuracy. The accuracy, I expect would still be within a passing range despite this nonetheless (perhaps an estimate of 70% or so).

+++

While I believe both the Gaussian Naive Bayes and Decision Tree models would be reasonably effective in our classification analysis, I would have reason to expect that the Gaussian Naive Bayes model would perform better. This is because the Gaussian Naive Bayes algorithm assumes that the data has a Gaussian distribution. While I am unsure that the data had this distribution, it is reasonable to assume that most large datasets with a plethora of continuous attributes would hold a Gaussian shape. While the Decision Tree would likely still be effective, it follows a flowchart-like pattern where it looks at each attribute one by one. Since some attributes seem to characterize breast cancer tumors with more statistical significance than others, it is reasonable to assume that Decision Tree may have trouble at these roadblocks. Since the Gaussian Naive Bayes algorithm is on the basis of a Gaussian distribution, which most large continuous datasets have, it is likely to be more effective for our purposes.

+++

## 7.3 - Basic Classification

```{code-cell} ipython3
feature_vars = ['Radius (mean)', 'Texture (mean)', 'Perimeter (mean)', 'Area (mean)', 'Smoothness (mean)', 'Compactness (mean)', 'Concavity (mean)', 'Concave points (mean)', 'Symmetry (mean)', 'Fractal dimension (mean)']
target_var = 'Diagnosis'
```

Here we split our data into X and y, where X represents only our feature variables and y represents our target variable (Diagnosis). We split each X and y further into training data and testing data so that a random 80% of our cancer_data is placed into training and the remaining 20% into testing.

```{code-cell} ipython3
X_train, X_test, y_train, y_test = train_test_split(cancer_df[feature_vars],
                                                    cancer_df[target_var],
                                                    test_size = 0.2,
                                                    random_state=17)
```

```{code-cell} ipython3
X_train.shape, X_test.shape
```

As we see, the train_test_split() function successfully separated out dataset into a training dataset (80%) and a testing dataset (20%)

```{code-cell} ipython3
X_train.head()
```

We can see that our training data successfully contains all of our characterizing attributes for our cancer tumors.

```{code-cell} ipython3
X_test.head()
```

We can see that our testing data also included only our characterizing attributes for our cancer tumors. This data is 20% of our original dataframe and will be used to test the accuracy of the Gaussian Naive Bayes algorithm.

```{code-cell} ipython3
gnb = GaussianNB()
```

```{code-cell} ipython3
gnb.fit(X_train,y_train)
```

```{code-cell} ipython3
gnb.__dict__
```

This fitted Gaussian Naive Bayes dictionary now contains many elements. The array labeled as 'theta_' contains the mean for each feature (Radius, Texture, Perimeter, etc.). The array labeled as 'var_' contains the standard deviation for each of these theta values.

```{code-cell} ipython3
gnb.score(X_test,y_test)
```

```{code-cell} ipython3
N = 20
n_features = len(feature_vars)
gnb_df = pd.DataFrame(np.concatenate([np.random.multivariate_normal(th, sig*np.eye(n_features),N)
                 for th, sig in zip(gnb.theta_,gnb.var_)]),
                 columns = gnb.feature_names_in_)
gnb_df['Diagnosis'] = [ci for cl in [[c]*N for c in gnb.classes_] for ci in cl]
sns.pairplot(data =gnb_df, hue='Diagnosis')
```

This model does seem to make sense as it is in alignment with my prediction. I assumed that some features like Radius and Perimeter of the cancer tumor would characterize malignant and benign classification better than other features. This is demonstrated by Gaussian curves based on the __dict__ parameter than do not overlap too significantly (though there still is an overlap). However, for some parameters like Texture and Fractal Dimension, there is an almost complete overlap. This demonstrates that some features are better for classification than others, suggesting that Gaussian Naive Bayes was a more effective choice when compared to the Decision Tree model.

+++

This suggests that the training data from the Gaussian Naive Bayes model does fit the data well. As there is almost a perfect match between our hypothesis and the synthetic data created by our gnb __dict__, this training data seems to resemble our original dataframe well. This means we can now use this training data to test our model and generate a classification report.

+++

We first generate our predicted dataset using our X_test

```{code-cell} ipython3
y_pred = gnb.predict(X_test)
```

Using out y_test and our y_pred, we can generate a confusion matrix where the indicies start from top-left to bottom-right as: <br> TN, FP <br> FN, TP

```{code-cell} ipython3
skmetrics.confusion_matrix(y_test,y_pred)
```

We then can organize this into a readable dataframe where each axis represents the predicted or true class (benign or malignant)

```{code-cell} ipython3
n_classes = len(gnb.classes_)
prediction_labels = [['predicted class']*n_classes, gnb.classes_]
actual_labels = [['true class']*n_classes, gnb.classes_]
conf_mat = skmetrics.confusion_matrix(y_test,y_pred)
conf_df = pd.DataFrame(data = conf_mat, index=actual_labels, columns=prediction_labels)
```

```{code-cell} ipython3
conf_df
```

Finally, we can generate a classification report, giving us useful information such as precision and recall

```{code-cell} ipython3
print(skmetrics.classification_report(y_test,y_pred))
```

After looking at final classification report of this model, I would recommend this model for deployment. While the model is not perfect, especially in specific features, there is an overall success with this model. For benign cancer tumors, the precision and recall of this model is 96%, meaning that the model correctly predicted benign tumors with significant agreement. While both precision and recall drop to 92% for malignant breast cancer tumors, this is still a very good statistical probability, further supporting the model's success. In the same range, there was a significant overall accuracy of 95% for the model's predictions. Based on these classification statistics, I would trust this model in deployment in the medical field.

+++

While a more complex model may deal with the overlapping Gaussian features better than this one, I would still support the use of this model for breast cancer tumor classification. With good precision, recall, and accuracy statistics, there is little reason to suggest that this model would not be effective in deployment. However, as the medical field is very precise, it would not hurt to use a more complex model if available. This evaluation has demonstrated that machine learning models like the Gaussian Naive Bayes would be effective at performing tasks like these. While this task is able to be done without the use of a machine learning model, by ultrasound sonographers for instance, the success of this evaluation would eagerly encourage more research with similar models and test cases in order to allow the widespread use of machine learning breast cancer tumor classification.

+++

## 7.4 - Exploring Problem Setups

```{code-cell} ipython3
percentages = [0.1, 0.3, 0.5, 0.7, 0.9]
train_percentages = []
n_train_samples = []
n_test_samples = []
train_accuracies = []
test_accuracies = []

for train_prct in percentages:
    X_train, X_test, y_train, y_test = train_test_split(cancer_df[feature_vars],
                                                        cancer_df[target_var],
                                                        train_size = train_prct,
                                                        random_state=17)
    
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    y_train_pred = gnb.predict(X_train)
    train_acc = skmetrics.accuracy_score(y_train, y_train_pred)

    y_test_pred = gnb.predict(X_test)
    test_acc = skmetrics.accuracy_score(y_test, y_test_pred)

    train_percentages.append(train_prct * 100)
    n_train_samples.append(n_train)
    n_test_samples.append(n_test)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

test_size_v_acc_df = pd.DataFrame({
    'train_pct': train_percentages,
    'n_train_samples': n_train_samples,
    'n_test_samples': n_test_samples,
    'train_acc': train_accuracies,
    'test_acc': test_accuracies })
    
```

```{code-cell} ipython3
test_size_v_acc_df
```

This dataframe shows how the accuracy in the training and test data changes as the training percentage increases. The training percentage models how much of the original dataframe is used to train the machine learning model while the test percentage is the remainder. We can observe that as the training percentage increases, the accuracy of the training model to classify benign versus malignant breast cancer tumors stays relatively stagnant. There seems to be no impact on the number of training samples on the training accuracy. This could be because the training accuracy is independent from the model itself. As the trained model is used for classification of test cases, we would expect that the testing accuracy increases increased as the training percentage increases. This is because more training samples results in more data to base testing classification on. This prediction is demonstrated by a consistently increasing test accuracy from ~92 - ~96%. While this data seems to support our hypothesis, it is worth noting that this would be more thoroughly examined using different random seeds, to see if this trend is consistent for different training and testing arrangements of the same percentage.

```{code-cell} ipython3

```
