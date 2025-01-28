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
# Basic Clustering

Basic clustering analysis using an open dataset on the activity of breast cancer cells (malignant/benign)


```{code-cell} ipython3
import seaborn as sns
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn import metrics
import pandas as pd
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

This is a dataset representing different case studies of breast cancer tumors. For each tumor, there is a designation on its activity (malignant 'M' and benign 'B'). There are columns in this dataset that represent quantitative measurements on the various properties of cell nucleis within each sample cancer tumor (ie. Radius, Texture, and Perimeter). This results in three sets of datasets (mean, standard deviation, and worst). These 'datasets' describe overall analysis of all sample cancer nucleis within a sample tumor (given by the ID number).

+++

This dataset will be analyzed using the KMeans clustering algorithm. The use of this algorithm will aim to properly characterize the differences in malignant and benign cancer tumors by separating the datasets into two clusters. While direct classification will not be performed from this algorithm for each sample, the prospect of two separate clusters demonstrates a difference between the properties of each diagnosis. If this clustering algorithm does not separate the dataset into two distinct clusters, resulting in significant overlap, then there would be no clear difference between the nucleis of malignant and benign cancer cells. This would provide valuable albeit disappointing insight on the ability of machine learning to distinguish the two diagnoses.

+++

While the classification task using this dataset in Assignment 7 sought to test the ability of the Naive Bayes Gaussian algorithm to correctly identify diagnoses based on previous training data, the clustering KMeans algorithm uses the assumption that the diagnosis is not known. As there is an extra variable inhibiting out approach to correctly identify malignant or benign tumors, the clustering approach allows us to still distinguish commonalities between the two tumors. In scenarios when there can not be training data (as the result ('y') is not known), clustering the dataset into groups can be useful for later analysis and identification.

```{code-cell} ipython3
cancer_df = cancer_df.iloc[:, :12]
sns.pairplot(cancer_df)
```

By plotting each attribute against each other using a series of scatterplots, we see that there seem to be no concrete groups between the two types of cancer tumors. This comparison of attributes is how the KMeans clustering algorithm will examine our dataset. This is why it is interesting to see if a clustering algorithm can make two reasonable groups out of a dataset which seem to have none from the human eye alone.

+++

Here we prepare our input dataset for clustering, by getting rid of the target column 'Diagnosis'. We also ensured above that our dataset only includes our mean data for separating into groups, so these groups are formed by similarities in the average measurement, with no influence on similarities of standard deviations or outlying 'worst' instances.

```{code-cell} ipython3
cancer_X = cancer_df.drop(columns=['Diagnosis', 'ID number'])
cancer_X.head()
```

Here we define our KMeans function with a target of 2 clusters. This is because we know there are supposed to be 2 clusters in the dataset, so we force the algorithm to 'find' 2 distinct clusters the best it can.

```{code-cell} ipython3
km2 = KMeans(n_clusters=2, random_state = 17)
```

Now we fit and predict our dataset using the KMeans algorithm

```{code-cell} ipython3
cancer_df['km2'] = km2.fit_predict(cancer_X)
```

## Evaluating Our Clusters

+++

### Visualization

+++

We will first do an evaluation of our clusters by graphically comparing our cancer cell attributes using the same pairplot as shown previously. The above pairplot contained only one color, demonstrating that there was no information on the activity of the cell (malignant or benign). After predicting our clusters with an 'n' value of 2, we can re-create our pairplot. This visualization shows two given colors modeling our clusters. For some attributes, the clusters seem relatively separated, meaning there were some distinguishable properties between each activity type. However, each cluster in this case border each other with little space between them, visually demonstrating that even the separable properties were not entirely distinct. Conversely, some attribute comparisons show no separation between clusters, with a complete overlap of predicted malignant and benign differences. This agrees with our initial expectation that some nuclei properties would not contribute positively to cluster separation.

```{code-cell} ipython3
sns.pairplot(data=cancer_df,hue='km2')
```

### Silhouette Score

+++

We can also evaluate our clustering quantitatively using a metric like the silhouette score. This score models both how far apart the clusters are from each other and how spread out each point is within a cluster. A low value (0) demonstrates that the clusters are not distinguishable from each other while the higher value (1) demonstrates that clusters and entirely separated and distinct. For our two clusters (characterizing malignant and benign nuclei properties), we see that there is a relatively high silhouette score of approximately 0.699. While this is still not a great silhouette score, it is better than expected and shows that two reasonable clusters were able to be created.

```{code-cell} ipython3
metrics.silhouette_score(cancer_X,cancer_df['km2'])
```

### Adjusted Mutual Info Score

+++

This next evaluation compares our clusters with the ground truth labels (the actual diagnosis of malignant or benign). We use the adjusted mutual info score algorithm, which essentially uses a determined correlation between two categorical variables to test the effectiveness of our clusters. The two parameters we give are our y_true (Actual Diagnosis) and our y_pred (KMeans Determination). We can see that our Adjusted Mutual Info Score is approximately 0.419. This is not a very good clustering score, suggesting that while the KMeans algorithm gave us two clusters, there was little consistency in whether or not the given prediction would effectively classify our Dianosis. While there was general structure that could allow for clustering, there was only moderate differentiation between actual malignant or benign cancer cell nucleis. In comparison to Assignment 7, while the classification accuracy was fairly good given a training dataset, the ability for differentiation without one is less promising.

```{code-cell} ipython3
metrics.adjusted_mutual_info_score(cancer_df['Diagnosis'], cancer_df['km2'])
```

## General Discussion

+++

As briefly discussed already, this clustering analysis has showed us that there was enough structure within the dataset to allow for separated clusters. While the clusters were visually adjoined in most attribute comparisons, they had a relatively good silhouette score. This initially suggested that the clustering was able to separate malignant and benign properties with a reasonable effectiveness. However, upon the examination of the adjusted mutual info score, comparing the clustering classifier with the actual diagnosis, it was shown that the clusters had only a moderate agreement with their identifier. This cluster evaluation means that while there was some structure within these clusters, there was only moderate success in differentiating actual malignant or benign cancer cells. As mentioned, the classification accuracy was relatively positive for a given training dataset in Assignment 7, while the ability for differentiation was less promising. Clustering performed worse than expected based on the high accuracy of classification, approximating at 95%. With this high of a classification success rate, it was expected that clustering would separate the two diagnoses well, which was not the case.

+++

## Further Experimentation with Different Cluster Sizes

+++

### Lower Cluster Size -- 1

```{code-cell} ipython3
km1 = KMeans(n_clusters = 1, random_state = 17)
cancer_df['km1'] = km1.fit_predict(cancer_X)
metrics.adjusted_mutual_info_score(cancer_df['Diagnosis'], cancer_df['km1'])
```

When we decrease the cluster size to 1, there is no meaningful correlation between the cluster and the diagnosis. This makes sense as nothing meaningful can come out of cluster size of 1. This would result in the data being a part of one distinct group, and therefore not able to be used to characterizing malignant versus benign cancer nuclei. This is cluster is therefore a conglomeration of the two previous clusters and is statistically useless in this analysis. When one clusters a dataset, they try to seek information that can distinguish the two groups without knowing what the specific attributes of a group may be. Finding similar patterns allows the user to find groups that may allow us to characterize each separately. When doing a cluster size of 1, this is not something that is possible and the point of the exercise is lost.

+++

### Higher Cluster Size -- 3

```{code-cell} ipython3
km3 = KMeans(n_clusters = 3, random_state = 17)
cancer_df['km3'] = km3.fit_predict(cancer_X)
metrics.adjusted_mutual_info_score(cancer_df['Diagnosis'], cancer_df['km3'])
```

```{code-cell} ipython3
metrics.silhouette_score(cancer_X,cancer_df['km3'])
```

```{code-cell} ipython3
sns.pairplot(data=cancer_df,hue='km3')
```

However, this becomes more interesting when we increase the cluster size to 3 (giving the algorithm another supposed option other than malignant or benign). What we see visually is that there is now a middle cluster that has been formed using data from both of the original 2 clusters. It is inserted the middle. We can see by doing this, the faults in the clustering algorithm. When an algorithm is forced to identify a certain amount of clusters, it will do so using the best patterns it can, often inventing new patterns to fit the request. This can be a problem in data science applications when the number of clusters is unknown. We see here that the silhouette score is actually higher than with only 2 clusters, suggesting that there was even better structure for the separation of 3 clusters. However, the adjusted mutual info score was lower, meaning that the despite the supporting structure, the information does not make much sense in a binary application such as this. This again enforces the worries in forcing more clusters than needed. While this can be a useful approach to distinguish sub-categories and properties within clusters, this application has no realistic purpose with this dataset.

```{code-cell} ipython3

```
