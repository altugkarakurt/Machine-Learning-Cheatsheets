<center><h1>Data Preprocessing</h1></center>

##Imputation:
Handling datapoints with missing columns.

Related libraries: [`sklearn.preprocessing.Imputer`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing) [`pandas.DataFrame.fillna`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.fillna.html)

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

## Standardization

## Polynomial Features

## Encoding Categorical Data