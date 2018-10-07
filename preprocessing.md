<center><h1>Data Preprocessing</h1></center>

## Imputation:
Handling datapoints with missing columns.

[`sklearn.preprocessing.SimpleImputer`](http://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html#sklearn.impute.SimpleImputer) [`pandas.DataFrame.fillna`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.fillna.html)

- **Listwise (Complete Case) Deletion:** Remove the rows that have missing columns.
  - Might introduce bias if there is an underlying pattern to the entries with missing values.
  - Shrinks the dataset.
  - Very easy to implement.

- **Pairwise Deletion:** Only delete the rows that have missing values on columns that are used in the particular analysis.
  - Presserves more datapoints than listwise deletion.
  - Leads the dataset size to be non-homogeneous across different analyses.

- **Hot Deck:** Filling the missing values with the values from a "similar" row. 
  - One variant is called "last observation carried forward" (LOCF), which assumes the dataset is sorted and fills in the missing value with the previous record's. This makes more sense in the case of time-series data, as it corresponds to falling back to prior observation, but doesn't make sense in independent observations of different entities, etc.

- **Cold Deck:** Fill the missing field from a similar entry from another dataset.

- **Mean Substitution:** Replace the missing value with the mean of the column across the dataset.
  - Preserves the mean of the value, but underestimates the correlations. This makes this method suitable for univariate analyses, not for multivariate ones.

- **Regression:** Using the rows with valid entries, fit a regression model to estimate the missing column from other columns.
  - If there are multiple missing columns in certain rows, things get tricky as you need to address some first and use that fabricated point to estimate the other.
  - Since the regression places the point perfectly fits the regression line, implying a correlation of 1 between the predictors and the missing outcome variable. This leads to overestimating the correlations.
  - One idea to account for this fabricated correlation is to use Stochastic Regression and add the average regression variance to introduce some error/noise.

- **'Missing' Category:** In the case of categorical features, we can generate a new category 'Missing' and impute this values to the missing entries. If there is a pattern or reason for these values to be missing, this category captures this phenomenon by explicitly marking them as missing.

- **Surrogate Split:** In the case of decision trees, if we know there are at most d features missing for a row, instead of using one split at every fork, we can generate a sorted list of d splits that we can fall back to in case of missing values. 

## Encoding Categorical Data
The main concern with mapping categories to numbers is the common implicit assumption in many models that the numerical values are algebraic. So, different numerical mappings would yield different results as the ordering of the target values might change.

- **One Hot Encoding:** Enumerating an n-categorical feature as a binary n-long vector. For each row, only one entry of this vector would be 1, at the index corresponding to the category it falls under. This approach transforms one categorical feature to n binary features.<br>[`sklearn.features_extraction.DictVectorizer`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html)

## Standardization
The main goal is to have zero mean and unit (order) variance in all features to prevent some features dominate the others.
[`sklearn.preprocessing.scale`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.scale.html#sklearn.preprocessing.scale)

## Polynomial Features
