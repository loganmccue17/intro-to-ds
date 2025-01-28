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

# Accuracy, Precision, and Recall
An analysis of basic confusion matrix metrics using open data on adult income.

```{code-cell} ipython3
import pandas as pd
from sklearn import metrics
import seaborn as sns
import numpy as np
my_num = 14
np.random.seed(my_num)
```

```{code-cell} ipython3
models_df = pd.read_csv('adult_models_only.csv')
```

```{code-cell} ipython3
models_to_audit = np.random.choice( models_df.columns,2)
```

```{code-cell} ipython3
models_to_audit
```

```{code-cell} ipython3
actual_df = pd.read_csv('adult_reconstruction_bin.csv')
```

```{code-cell} ipython3
actual_df.columns.values[0] = 'id'
```

```{code-cell} ipython3
merged_df = pd.merge(models_df, actual_df, how = 'inner', on = 'id')
merged_df.columns
```

## Accuracy for GPR income>=20k

```{code-cell} ipython3
acc_GPR_20k_fx  = lambda d: metrics.accuracy_score(d['income>=20k'],d[models_to_audit[0]])
acc_GPR = merged_df.groupby('gender').apply(acc_GPR_20k_fx)
```

## Recall for GPR income>=20k

```{code-cell} ipython3
recall_GPR_20k_fx  = lambda d: metrics.recall_score(d['income>=20k'],d[models_to_audit[0]])
recall_GPR = merged_df.groupby('gender').apply(recall_GPR_20k_fx)
```

## Precision for GPR income>=20k

```{code-cell} ipython3
prec_GPR_20k_fx  = lambda d: metrics.precision_score(d['income>=20k'],d[models_to_audit[0]])
prec_GPR = merged_df.groupby('gender').apply(prec_GPR_20k_fx)
```

## Accuracy for RPR income>= 60k

```{code-cell} ipython3
acc_RPR_60k_fx  = lambda d: metrics.accuracy_score(d['income>=60k'],d[models_to_audit[1]])
acc_RPR = merged_df.groupby('gender').apply(acc_RPR_60k_fx)
```

## Recall for RPR income>=60k

```{code-cell} ipython3
recall_RPR_60k_fx  = lambda d: metrics.recall_score(d['income>=60k'],d[models_to_audit[1]])
recall_RPR = merged_df.groupby('gender').apply(recall_RPR_60k_fx)
```

## Precision for RPR income>=60k

```{code-cell} ipython3
prec_RPR_60k_fx  = lambda d: metrics.precision_score(d['income>=60k'],d[models_to_audit[1]])
prec_RPR = merged_df.groupby('gender').apply(prec_RPR_60k_fx)
```

```{code-cell} ipython3
pd.DataFrame({'Accuracy GPR>=20k' : acc_GPR, 'Recall GPR>=20k' : recall_GPR, 'Precision GPR>=20k' : prec_GPR, 'Accuracy RPR>=60k' : acc_RPR, 'Recall RPR>=60k' : recall_RPR, 'Precision RPR>=60k' : prec_RPR})
```

Accuracy: The ratio of correctly predicted instances (both true positives and true negatives) to the total instances. <br>
Recall: The ratio of correctly predicted positive instances to the actual total positive instances. <br>
Precision: The ratio of correctly predicted positive instances to the total predicted positive instances.

+++

## Accuracy, Recall, and Precision

+++

Looking at these tables we can see the GPR model predicted males more accurately in terms of their income. This was a resultant accuracy of roughly 79% compared to 70% for females. This may demonstrate a slight bias towards male prediction, if outside of error bars. This is also for a lower income threshold, meaning at these low incomes, men were predicted more on average than women. This is different when compared to the RPR machine learning model at a threshold of 60k or above. In this scenario, females were accurately predicted by almost 95%, when compared to only 82% for males. This is significant, and demonstrates that at higher incomes, the model likely had a negative bias towards men. However, under both instances, men and female were predicted more accurately under the RPR model. This likely supports in the use of the RPR model despite a potential gender bias, however it would be best to compare the machine learning models at the same income threshold to determine if the biasy is more related to the model or the threshold.

+++

When looking at recall, we see more significant results. The recall using the GPR model demonstrates that only about 58% of females that WERE in this income threshold were predicted correctly. This is compared to males at almost 90%. These drastic results characterize a notable biasy in the GPR machine learning model. At a lower income threshold, the model is more apt to predict correctly a male that works at this level. This seems to be a biasy against female prediction, meaning that the model may women with a lower income, when that may not be the case. However, to see this biasy more clearly, we would have to look at a higher income level also. This would show if it is a model biasy or an income biasy (or both). The RPR model, however, does not seem to have a significant biasy between males and females for a higher 60k+ income level. However, the model also performs quite poorly at correctly identifying people in general who do indeed work for a high income. So despite the lack of a gender biasy (or an insignificant one) at this threshold for RPR, the model was still not strong at this income level.

+++

Precision similarly measured a ratio of predicted positive instances. However while recall looks at how many positive instances were correctly predicted positive, precision looks at how many predicted positive instances were correct. What we can see through our analysis is that for the GPR machine learning model at a threshold of >=20k, about two-thirds of the model's positive predictions were actually positive for females. This number jumps to around 81% for males. This means that the model predicted excess females as being within the lower income threshold then men, meaning the model may have a biasy towards women working lower income jobs. This is compared with the RPR model which predicts about 75% for females and 70% for males. This data characterizes a slight biasy towards females working higher income jobs, seemingly the opposite. Despite the potential biasy, it is hard to characterize this to its fullest extent without comparing equivalent income ranges between the models.

+++

## Which To Deploy?

+++

Looking between these two models, I would say that the RPR model is the best to implement. The model demonstrates good accuracy and precision without significant gender bias. While the bias may still be present, it is difficult to know without examining other income levels, against more examples of a counter model, or without knowing specific standard deviations. While the RPR model had poor recall for both men and women, this may be an effect of working with high incomes levels in general (and not a specific problem with the model itself). The GPR however demonstrates significant gender bias against women, consistently predicting that they work for a lower income incorrectly.

```{code-cell} ipython3

```
