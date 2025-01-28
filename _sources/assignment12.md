# Text Vectorization

An classification of fake and real news articles using techniques of text vectorization. This serves as an introduction to machine learning based on text.

## Classification of Fake News

First we import all of our metrics and functions that we will need for this analysis. This includes our standard libraries, our euclidean vector analysis tool, and our classification methods.


```python
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
```

We then import our dataset in its original form from the given csv file. This will be stored as news_df.


```python
news_df = pd.read_csv('fake_or_real_news.csv')
news_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>title</th>
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8476</td>
      <td>You Can Smell Hillary’s Fear</td>
      <td>Daniel Greenfield, a Shillman Journalism Fello...</td>
      <td>FAKE</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10294</td>
      <td>Watch The Exact Moment Paul Ryan Committed Pol...</td>
      <td>Google Pinterest Digg Linkedin Reddit Stumbleu...</td>
      <td>FAKE</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3608</td>
      <td>Kerry to go to Paris in gesture of sympathy</td>
      <td>U.S. Secretary of State John F. Kerry said Mon...</td>
      <td>REAL</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10142</td>
      <td>Bernie supporters on Twitter erupt in anger ag...</td>
      <td>— Kaydee King (@KaydeeKing) November 9, 2016 T...</td>
      <td>FAKE</td>
    </tr>
    <tr>
      <th>4</th>
      <td>875</td>
      <td>The Battle of New York: Why This Primary Matters</td>
      <td>It's primary day in New York and front-runners...</td>
      <td>REAL</td>
    </tr>
  </tbody>
</table>
</div>



We find the shape of the dataframe to find how many articles are in this dataset (6335).


```python
news_df.shape
```




    (6335, 4)



We separate this data into three new datasets. titles_X includes all of the titles of our news articles in the same order as the original dataset. texts_X includes all of the texts of our news articles in the same order. news_y takes in the actual label given to the dataset for training purposes (Fake or Real).


```python
titles_X = news_df['title']
texts_X = news_df['text']
news_y = news_df['label']
```

While scikit-learn will be able to deal with labels directly like REAL and FAKE, we convert these labels to a binary 0 and 1 to easily understand this conversion (instead of determining later what the algorithm had chosen for 0 and 1).


```python
news_y = news_y.replace({'FAKE' : 0, 'REAL' : 1})
```

    /var/folders/xf/qc8rdk_57079zlfj9cc9fqvw0000gn/T/ipykernel_5440/2215345829.py:1: FutureWarning: Downcasting behavior in `replace` is deprecated and will be removed in a future version. To retain the old behavior, explicitly call `result.infer_objects(copy=False)`. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
      news_y = news_y.replace({'FAKE' : 0, 'REAL' : 1})


Here we convert our list of titles and texts to a vector matrix. This vector matrix represents the patterns of words in these articles.


```python
count_vec = text.CountVectorizer()
titles_X_vec = count_vec.fit_transform(titles_X)
texts_X_vec = count_vec.fit_transform(texts_X)
```

We can analyze the result of this through the shape attribute of our vector matrix. The first number (# of rows) tells us how many articles are present and is equal to the number of rows in the original dataframe. The second number (# of cols) tells us how many unique words are present among all article titles (or texts respectively). Each value in our matrix will represent a count of the occurences of each unique word in each article.


```python
titles_X_vec.shape, texts_X_vec.shape
```




    ((6335, 10071), (6335, 67659))



Before we split into testing and training data, we will convert our vectors to another format. Here we convert our vectors into TD-IDF, which keeps into account not only word frequency, but the frequency among our total documents. This will allow us to compare vocabulary representations and optimize to see which is better suited for our classifier.


```python
tfidf = text.TfidfTransformer()
titles_X_tfidf = tfidf.fit_transform(titles_X_vec)
texts_X_tfidf = tfidf.fit_transform(texts_X_vec)
```

We can notice that the tfidf matrix has the same shape as the vector matricies for both titles and texts. This is because the rows and cols are similar, but with an extra parameter in its subsequent calculation.


```python
titles_X_tfidf.shape
```




    (6335, 10071)




```python
texts_X_tfidf.shape
```




    (6335, 67659)



Here we split our **titles_X_vec** representation and our **titles_X_tfidf** representation into training and testing data, alongside the usual y output. We do this all at once for our titles and our texts in order to compare more fairly.


```python
titles_vec_train, titles_vec_test, texts_vec_train, texts_vec_test, titles_tfidf_train, titles_tfidf_test, texts_tfidf_train, texts_tfidf_test, y_train, y_test = train_test_split(titles_X_vec, texts_X_vec, titles_X_tfidf, texts_X_tfidf, news_y, random_state = 17)
```

We then instantiate our classification models, one each for the vec and tfidf representations.


```python
clf_vec_titles = MultinomialNB()
clf_vec_texts = MultinomialNB()
clf_tfidf_titles = GaussianNB()
clf_tfidf_texts = GaussianNB()
```

This next step fits out classifier model with the titles vector training data, and subsequently scores the model. As usual, the fitting is done with X and y training data, while the scoring is done with X and y testing data.

#### Titles Vector Representation Score


```python
clf_vec_titles.fit(titles_vec_train, y_train).score(titles_vec_test, y_test)
```




    0.8251262626262627



#### Texts Vector Representation Score


```python
clf_vec_texts.fit(texts_vec_train, y_train).score(texts_vec_test, y_test)
```




    0.8813131313131313



#### Titles TF-IDF Representation Score


```python
clf_tfidf_titles.fit(titles_tfidf_train.toarray(), y_train).score(titles_tfidf_test.toarray(), y_test)
```




    0.6849747474747475



#### Texts TF-IDF Representation Score


```python
clf_tfidf_texts.fit(texts_tfidf_train.toarray(), y_train).score(texts_tfidf_test.toarray(), y_test)
```




    0.7853535353535354



After fitting and scoring our four scenarios (texts and titles, both with a vector and tf-idf representation, we have a preliminary observation that the Texts of an article can be the most predictive of its realness or fakeness. However, this is only characterizable in a vector representation of its vocabulary, and NOT with the tf-idf representation, which is surprising.

While these are interesting results, it would be more valuable if we performed a cross-validation analysis. This is because these scores are random based on the state in which we split our data into training and testing dataframes. A cross validation will give us a chance to find an average of multiple splits. I have added the import at the beginning of the mark down so we can use a simple cross validation.

We have to use our vector and tfidf dataframes, BEFORE we split into training and testing data, so we pass in the complete titles_X_vec, et cetera. Because the dataset is quite large, we can use a larger amount of folds that give us an interesting representation.

#### Titles Vector Cross-Validation Score


```python
np.mean(cross_val_score(clf_vec_titles, titles_X_vec, news_y, cv = 10))
```




    0.8209906259811323



#### Texts Vector Cross-Validation Score


```python
np.mean(cross_val_score(clf_vec_texts, texts_X_vec, news_y, cv = 10))
```




    0.8865023597011877



#### Titles TF-IDF Cross-Validation Score


```python
np.mean(cross_val_score(clf_tfidf_titles, titles_X_tfidf.toarray(), news_y, cv = 10))
```




    0.6915541634896667



#### Texts TF-IDF Cross-Validation Score


```python
np.mean(cross_val_score(clf_tfidf_texts, texts_X_tfidf.toarray(), news_y, cv = 10))
```




    0.8050453252002132



These results are very similar to our results for one specific training and testing set, but now we have proved that it remains consistent for an average across different splits. We can say with much better certainty that while all scenarios are fairly good at model this prediction, the prediction is more effective using article texts with a vector representation.

Based on our above analysis, we have determined that texts are more predictive of the legitimacy of an article, however only for a certain representation. We have tested the titles and the texts of an article separately with two different representations, a count-vector matrix and a term frequency-inverse document frequency (tfidf) vector matrix. Among these four scenarios total, we have found that the highest predictive score (after a 10-fold cross validation) was the article texts as a count-vector matrix. This had an accuracy score of about 0.88 on average. Given that this analysis was done over a 10-fold cross validation, I would suggest that there is enough evidence to support this claim, as this was a much higher score that is repeatable over a computer average. While the text count-vectorization representation is the most predictive, we see that under a tf-idf comparison, it is much lower. On a 10-fold average, we see that the tf-idf performs at about 80% success for text, significantly lower that the count-vector's 88%. Surprisingly at first look, the title performed at about 82% in the vector representation, but much lower at about 69% under the TF-IDF representation and the Gaussian Naive Bayes Model. I would expect that this is because of the lack of a vocabulary in titles, meaning simply less words to aid in prediction. This would also explain the lesser score for the count-vector representation.

We can look at both the title the text analysis to find which representation is better. This ends up being the count-vector, with the optimal score of 0.88 for the vector when compared to 0.80 for the tf-idf representation. For the title analysis, this is even greater, where we see a roughly 0.82 score for the vector over 0.69 for the tf-idf. And with that comparison, we can further say that not only is the text of an article more predictive of an article's legitimacy, but the count vector representation is better for model training than the tf-idf. This could also further conclude, that not only is the count vector representation better than the tf-idf representation, but the Multinomial Naive Bayes model may be more effective than the Gaussian Naive Bayes model for this analysis. While Gaussian Naive Bayes is used to better utilize the TF-IDF representation, it is still not as effective as the use of the Multinomial Naive Bayes model for the count-vector representation.

## Model Optimization

We will then optimize our best model, to determine if we can achive an optimal cross validation score. The model we will use is our article texts in a count-vector representation with a Multinomial Naive Bayes model.


```python
clf_vec_texts = MultinomialNB()
```

We choose to increment alpha by factors of 10 in order to get a general range of where this parameter should lie.


```python
param_dt = {'alpha': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100,1000]}
```

Using our vec_texts classifer (MultinomialNB), we optimize the model with our parameters and get a resultant GridSearch with a cross-validation of 10.


```python
clf_vec_texts_opt = GridSearchCV(clf_vec_texts, param_dt, cv = 10)
```


```python
clf_vec_texts_opt.fit(texts_X_vec, news_y)
```




<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=10, estimator=MultinomialNB(),
             param_grid={&#x27;alpha&#x27;: [1e-05, 0.0001, 0.001, 0.1, 1, 10, 100,
                                   1000]})</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;GridSearchCV<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.GridSearchCV.html">?<span>Documentation for GridSearchCV</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>GridSearchCV(cv=10, estimator=MultinomialNB(),
             param_grid={&#x27;alpha&#x27;: [1e-05, 0.0001, 0.001, 0.1, 1, 10, 100,
                                   1000]})</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">best_estimator_: MultinomialNB</label><div class="sk-toggleable__content fitted"><pre>MultinomialNB(alpha=0.0001)</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;MultinomialNB<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.naive_bayes.MultinomialNB.html">?<span>Documentation for MultinomialNB</span></a></label><div class="sk-toggleable__content fitted"><pre>MultinomialNB(alpha=0.0001)</pre></div> </div></div></div></div></div></div></div></div></div>



We find that the most optimized parameters of alpha is 0.0001


```python
clf_vec_texts_opt.best_params_
```




    {'alpha': 0.0001}



We then score our optimize model to see how successful it is in determining the legitimacy of our articles. We find that this optimized model gives a score of 0.974, which is very good! Because of this cross-validation score, I would absolutely recommend the deployment of this model and would be confident saying that the count vectorization representation of article texts under a Multnomial Naive Bayes classification model is the most effective in determining an article legitimacy.


```python
clf_vec_texts_opt.score(texts_vec_test, y_test)
```




    0.9741161616161617



## Data Visualization

For this next step, we will then try to visualize our data using a heatmap. We first grab the indices respectively where news_y correlates to fake news, and then real news. We concatenate so we can have all fake news articles listed first, and then all real news articles.


```python
fake_indices = np.where(news_y == 0)[0]
real_indices = np.where(news_y == 1)[0]
subset_rows = np.concatenate([fake_indices, real_indices])
```

Here we grab the lengths of our fake_indices and real_indices to make sure that our heatmap will more or less contain even quadrants.


```python
len(fake_indices), len(real_indices)
```




    (3164, 3171)



The number of fake articles to real articles is similar, which is good for a heatmap representation, giving us four distinct quadrants.

Doing this separation and subsequent concatenation allows us to model our heatmap with distinct quadrants. In the top-left quadrant we compare fake news articles with other fake news articles. The bottom-right quadrant compares real news articles with other real news articles. The other two quadrants compares real and fake news articles with each other. We perform this on both the titles and texts of the tfidf representation as it is not an effective visualization for count vector representations.


```python
sns.heatmap( euclidean_distances(titles_X_tfidf[subset_rows]))
```




    <Axes: >




    
![png](assignment12_files/assignment12_66_1.png)
    



```python
sns.heatmap( euclidean_distances(texts_X_tfidf[subset_rows]))
```




    <Axes: >




    
![png](assignment12_files/assignment12_67_1.png)
    


When looking at the heat maps, we see that titles are not a good indicator of different in their euclidean distances as representations are fairly homogenous. However, when looking at texts_tfidfs, we see that there is a general darker color in the bottom right quadrant. This means that real news articles are more similar in their vocabulary when compared to fake news, which tends to use much more varied terms.


```python
selected_categories = ['Fake', 'Real']
texts_y_pred = clf_tfidf_texts.predict(texts_tfidf_test.toarray())
cols = pd.MultiIndex.from_arrays([['actual class']*len(selected_categories),
                                  selected_categories])
rows = pd.MultiIndex.from_arrays([['predicted class']*len(selected_categories),
                                  selected_categories])

pd.DataFrame(data = confusion_matrix(texts_y_pred, y_test),columns = cols,
             index = rows)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="2" halign="left">actual class</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>Fake</th>
      <th>Real</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">predicted class</th>
      <th>Fake</th>
      <td>585</td>
      <td>149</td>
    </tr>
    <tr>
      <th>Real</th>
      <td>191</td>
      <td>659</td>
    </tr>
  </tbody>
</table>
</div>



This falls in line with our general analysis from the heat map. Since fake news articles have more varied vocabularies when compared to real news (albeit slightly), we would expect that the classifer would predict fakes news wrongly more than real news articles. As expected, this is demonstrated by our confusion matrix. We see that fake news was predicted as being real about 190 times. Real news however was predicted as being fake just under 150 times. While these values are fairly similar, the difference is still modeled and agrees with the heat map.

The heatmap given above contains four quadrants in which each is a comparison of either real to real, fake to fake, or across distinctions. We see that the vocabularies are quite varied across titles, as expected, regardless of its legitimacy. However, among article texts, we can observe that fake articles have a more varied vocabulary than real articles. This varied vocabulary in fake articles contributes to misclassification as determined by the confusion matrix in which more fake articles were classified as real than vice versa. This means that texts of real articles are more similar to each other, giving more room for fake articles to 'mimic' their vocabulary. Titles, however, seem to be different for more fake and real articles, making it hard to use that as a classifier (hence the lower success rate in cross validation).

We then decide to calculate the average euclidean distance for each quadrant and display the data in a pandas DataFrame. We do this to determine if there is noticeable significance in the variance of vocabularies among the types of articles. We have assumed that there is significant difference, but this will allow us to determine what exactly that average distance is.


```python
distances = euclidean_distances(titles_X_tfidf[subset_rows])

fake_v_fake = np.mean(distances[:len(fake_indices), :len(fake_indices)])
real_v_fake = np.mean(distances[len(fake_indices) + 1 :, :len(fake_indices)])
fake_v_real = np.mean(distances[:len(fake_indices), len(fake_indices) + 1 :])
real_v_real = np.mean(distances[len(fake_indices) + 1: , len(fake_indices) + 1:])

pd.DataFrame (data = [[fake_v_fake, real_v_fake], [fake_v_real, real_v_real]], columns = ['Fake', 'Real'], index = ['Fake', 'Real'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fake</th>
      <th>Real</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Fake</th>
      <td>1.405605</td>
      <td>1.406904</td>
    </tr>
    <tr>
      <th>Real</th>
      <td>1.406904</td>
      <td>1.404685</td>
    </tr>
  </tbody>
</table>
</div>



It turns out there actually is not much of a difference between the distances of fake/fake or real/real article relationships. The average difference between tfidf vocaularies of these articles are similar regardless of their legitimacy identification. This would explain why the TF-IDF did not perform as effectively as the vector representation, as their adjusted vocabulary representations are not very different across articles.

Finally, to show the similarity between the distances of these quadrants, we will plot a simple matplot. Here we have the type of euclidean distance comparison on the x-axis, and the average distance between article of the y-axis. We start by organizing the dataframe in a different format for better accessibility in seaborn.


```python
mean_distances = pd.DataFrame( data = [fake_v_fake, real_v_fake, fake_v_real, real_v_real], columns = ['Mean Euclidean Distance'], index = ['Fake v Fake', 'Real v Fake', 'Fake v Real', 'Real v Real'])
mean_distances
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mean Euclidean Distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Fake v Fake</th>
      <td>1.405605</td>
    </tr>
    <tr>
      <th>Real v Fake</th>
      <td>1.406904</td>
    </tr>
    <tr>
      <th>Fake v Real</th>
      <td>1.406904</td>
    </tr>
    <tr>
      <th>Real v Real</th>
      <td>1.404685</td>
    </tr>
  </tbody>
</table>
</div>



We then create the seaborn plot.


```python
md_plot = sns.barplot(mean_distances, x = mean_distances.index, y = 'Mean Euclidean Distance', hue = "Mean Euclidean Distance", legend = False)
for i in md_plot.containers:
    md_plot.bar_label(i,)
```


    
![png](assignment12_files/assignment12_78_0.png)
    


As we can see, the mean euclidean distance between the types of articles are almost equivalent, suggesting a similarity in TF-IDF representations which may be contributing to their lesser scores.


```python

```
