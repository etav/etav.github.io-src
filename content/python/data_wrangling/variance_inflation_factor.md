---
Title: Variance Inflation Factor (VIF) Explained
Slug: vif_factor_python
Summary: Variance Inflation Factor (VIF) Explained
Date: 2017-3-8 6:20
Category: Python
Tags: Data Wrangling
Authors: Ernest Tavares III
---

Colinearity is the state where two variables are highly correlated and contain similiar information about the variance within a given dataset. To detect colinearity among variables, simply create a correlation matrix and find variables with large absolute values. In R use the [`corr`](http://www.sthda.com/english/wiki/correlation-matrix-a-quick-start-guide-to-analyze-format-and-visualize-a-correlation-matrix-using-r-software) function and in python this can by accomplished by using numpy's [`corrcoef`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html) function.

[Multicolinearity](https://en.wikipedia.org/wiki/Multicollinearity) on the other hand is more troublesome to detect because it emerges when three or more variables, which are highly correlated, are included within a model. To make matters worst multicolinearity can emerge even when isolated pairs of variables are not colinear.


A common R function used for testing regression assumptions and specifically multicolinearity is "VIF()" and unlike many statistical concepts, its formula is straightforward:

$$ V.I.F. = 1 / (1 - R^2). $$

The Variance Inflation Factor (VIF) is a measure of colinearity  among predictor variables within a multiple regression. It is calculated by taking the the ratio of the variance of all a given model's betas divide by the variane of a single beta if it were fit alone.

### Steps for Implementing VIF
1. Run a multiple regression.
2. Calculate the VIF factors.
3. Inspect the factors for each predictor variable, if the VIF is between 5-10, multicolinearity is likely present and you should consider dropping the variable.


```python
#Imports
import pandas as pd
import numpy as np
from patsy import dmatrices
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

df = pd.read_csv('loan.csv')
df.dropna()
df = df._get_numeric_data() #drop non-numeric cols

df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>member_id</th>
      <th>loan_amnt</th>
      <th>funded_amnt</th>
      <th>funded_amnt_inv</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>annual_inc</th>
      <th>dti</th>
      <th>delinq_2yrs</th>
      <th>...</th>
      <th>total_bal_il</th>
      <th>il_util</th>
      <th>open_rv_12m</th>
      <th>open_rv_24m</th>
      <th>max_bal_bc</th>
      <th>all_util</th>
      <th>total_rev_hi_lim</th>
      <th>inq_fi</th>
      <th>total_cu_tl</th>
      <th>inq_last_12m</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1077501</td>
      <td>1296599</td>
      <td>5000.0</td>
      <td>5000.0</td>
      <td>4975.0</td>
      <td>10.65</td>
      <td>162.87</td>
      <td>24000.0</td>
      <td>27.65</td>
      <td>0.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1077430</td>
      <td>1314167</td>
      <td>2500.0</td>
      <td>2500.0</td>
      <td>2500.0</td>
      <td>15.27</td>
      <td>59.83</td>
      <td>30000.0</td>
      <td>1.00</td>
      <td>0.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1077175</td>
      <td>1313524</td>
      <td>2400.0</td>
      <td>2400.0</td>
      <td>2400.0</td>
      <td>15.96</td>
      <td>84.33</td>
      <td>12252.0</td>
      <td>8.72</td>
      <td>0.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1076863</td>
      <td>1277178</td>
      <td>10000.0</td>
      <td>10000.0</td>
      <td>10000.0</td>
      <td>13.49</td>
      <td>339.31</td>
      <td>49200.0</td>
      <td>20.00</td>
      <td>0.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1075358</td>
      <td>1311748</td>
      <td>3000.0</td>
      <td>3000.0</td>
      <td>3000.0</td>
      <td>12.69</td>
      <td>67.79</td>
      <td>80000.0</td>
      <td>17.94</td>
      <td>0.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 51 columns</p>
</div>




```python
df = df[['annual_inc','loan_amnt', 'funded_amnt','annual_inc','dti']].dropna() #subset the dataframe
```

## Step 1: Run a multiple regression


```python
%%capture
#gather features
features = "+".join(df.columns - ["annual_inc"])

# get y and X dataframes based on this regression:
y, X = dmatrices('annual_inc ~' + features, df, return_type='dataframe')
```

## Step 2: Calculate VIF Factors


```python
# For each X, calculate VIF and save in dataframe
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
```

## Step 3: Inspect VIF Factors


```python
vif.round(1)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VIF Factor</th>
      <th>features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>Intercept</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>dti</td>
    </tr>
    <tr>
      <th>2</th>
      <td>678.4</td>
      <td>funded_amnt</td>
    </tr>
    <tr>
      <th>3</th>
      <td>678.4</td>
      <td>loan_amnt</td>
    </tr>
  </tbody>
</table>
</div>



As expected, the total funded amount for the loan and the amount of the loan have a high variance inflation factor because they "explain" the same variance within this dataset. We would need to discard one of these variables before moving on to model building or risk building a model with high multicolinearity.
