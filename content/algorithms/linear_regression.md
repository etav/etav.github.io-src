---
Title: Implementing Linear Regression using Stats Models
Slug: linear_regression
Date: 2017-06-23 5:20
Category: Algorithms
Author: Ernest Tavares III
---

## Introduction to Linear Regression
Linear Regression or Ordinary Least Squares Regression (OLS) is one of the simplest machine learning algorithms and produces both accurate and interpretable results on most types of continuous data. While more sophisticated algorithms like random forest will produce more accurate results, they are know as [“black box”](http://machinelearningmastery.com/the-seductive-trap-of-black-box-machine-learning/) models because it’s tough for analysts to interpret the model. In contrast, OLS regression results are clearly interpretable because each predictor value (beta) is assigned a numeric value (coefficient) and a measure of significance for that variable (p-value). This allows the analyst to interpret the effect of difference predictors on the model and tune it easily.

Here we’ll use [college admissions data](https://collegescorecard.ed.gov/data/documentation/) and the `statsmodels` package to perform a simple linear regression looking at the relationship between average SAT score, out-of-state tuition and the selectivity for a range of US higher education institutions. We'll read the data using `pandas` and represent it visually using `matplotlib`.


```python
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
%matplotlib inline

cols = ['ADM_RATE','SAT_AVG', 'TUITIONFEE_OUT'] #cols to read, admit rate, avg sat score & out-of-state tuition

df = pd.read_csv('college_stats.csv', usecols=cols)
df.dropna(how='any', inplace=True)
len(df) #1303 schools
```




    1303



## Represent the OLS Results Numerically


```python
#fit X & y
y,X=(df['TUITIONFEE_OUT'], df[['SAT_AVG','ADM_RATE']])

#call the model
model = sm.OLS(y, X)

#fit the model
results = model.fit()

#view results
results.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>     <td>TUITIONFEE_OUT</td>  <th>  R-squared:         </th> <td>   0.919</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.919</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   7355.</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sat, 24 Jun 2017</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>12:11:48</td>     <th>  Log-Likelihood:    </th> <td> -13506.</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>  1303</td>      <th>  AIC:               </th> <td>2.702e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  1301</td>      <th>  BIC:               </th> <td>2.703e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     2</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>        <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th> <th>[95.0% Conf. Int.]</th>
</tr>
<tr>
  <th>SAT_AVG</th>  <td>   29.8260</td> <td>    0.577</td> <td>   51.699</td> <td> 0.000</td> <td>   28.694    30.958</td>
</tr>
<tr>
  <th>ADM_RATE</th> <td>-9600.6540</td> <td>  907.039</td> <td>  -10.585</td> <td> 0.000</td> <td>-1.14e+04 -7821.235</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 9.845</td> <th>  Durbin-Watson:     </th> <td>   1.313</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.007</td> <th>  Jarque-Bera (JB):  </th> <td>   7.664</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.090</td> <th>  Prob(JB):          </th> <td>  0.0217</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.670</td> <th>  Cond. No.          </th> <td>4.55e+03</td>
</tr>
</table>



Note that although we only used two variables we get a strong R-squared. This means that much of the variability within the out-of-state tuition can be explained or captured by SAT scores and selectivity or admittance rate.

## Represent the OLS Results Visually

### Plot of Out of State Tuition and Average SAT Score


```python
fig, ax = plt.subplots()
fig = sm.graphics.plot_fit(results, 0, ax=ax)
ax.set_ylabel("Out of State Tuition")
ax.set_xlabel("Avg SAT Score")
ax.set_title("OLS Regression")
```




    <matplotlib.text.Text at 0x10b6cd790>




![png](/images/linear_regression/tuition_sat.png)


### Plot of Out of State Tuition and Admittance Rate


```python
fig, ax = plt.subplots()
fig = sm.graphics.plot_fit(results, 1, ax=ax)
ax.set_ylabel("Out of State Tuition")
ax.set_xlabel("Admittance Rate")
ax.set_title("OLS Regression")
```




    <matplotlib.text.Text at 0x10cfd4390>




![png](/images/linear_regression/tuition_admit.png)
