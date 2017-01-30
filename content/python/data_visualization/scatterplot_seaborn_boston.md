---
Title: Scatter Plot in Python using Seaborn
Slug: scatter_plot_python_seaborn
Summary: Scatter Plot in Python using Seaborn
Date: 2017-1-29 7:43
Category: Python
Tags: Data Visualization
Authors: Ernest Tavares III
---

# Scatter Plot using Seaborn
One of the handiest visualization tools for making quick inferences about relationships between variables is the scatter plot. We're going to be using [Seaborn](http://seaborn.pydata.org/) and the boston housing data set from the [Sci-Kit Learn library](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html) to accomplish this.


```python
import pandas as pd
import seaborn as sb
%matplotlib inline
from sklearn import datasets
import matplotlib.pyplot as plt

sb.set(font_scale=1.2, style="ticks") #set styling preferences
dataset = datasets.load_boston()

#convert to pandas data frame
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['target'] = dataset.target
df.head()
df = df.rename(columns={'target': 'median_value', 'oldName2': 'newName2'})
df.DIS = df.DIS.round(0)
```

## Describe the data


```python
df.describe().round(1)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>median_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>506.0</td>
      <td>506.0</td>
      <td>506.0</td>
      <td>506.0</td>
      <td>506.0</td>
      <td>506.0</td>
      <td>506.0</td>
      <td>506.0</td>
      <td>506.0</td>
      <td>506.0</td>
      <td>506.0</td>
      <td>506.0</td>
      <td>506.0</td>
      <td>506.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.6</td>
      <td>11.4</td>
      <td>11.1</td>
      <td>0.1</td>
      <td>0.6</td>
      <td>6.3</td>
      <td>68.6</td>
      <td>3.8</td>
      <td>9.5</td>
      <td>408.2</td>
      <td>18.5</td>
      <td>356.7</td>
      <td>12.7</td>
      <td>22.5</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.6</td>
      <td>23.3</td>
      <td>6.9</td>
      <td>0.3</td>
      <td>0.1</td>
      <td>0.7</td>
      <td>28.1</td>
      <td>2.1</td>
      <td>8.7</td>
      <td>168.5</td>
      <td>2.2</td>
      <td>91.3</td>
      <td>7.1</td>
      <td>9.2</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.4</td>
      <td>3.6</td>
      <td>2.9</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>187.0</td>
      <td>12.6</td>
      <td>0.3</td>
      <td>1.7</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.1</td>
      <td>0.0</td>
      <td>5.2</td>
      <td>0.0</td>
      <td>0.4</td>
      <td>5.9</td>
      <td>45.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>279.0</td>
      <td>17.4</td>
      <td>375.4</td>
      <td>7.0</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.3</td>
      <td>0.0</td>
      <td>9.7</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>6.2</td>
      <td>77.5</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>330.0</td>
      <td>19.0</td>
      <td>391.4</td>
      <td>11.4</td>
      <td>21.2</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.6</td>
      <td>12.5</td>
      <td>18.1</td>
      <td>0.0</td>
      <td>0.6</td>
      <td>6.6</td>
      <td>94.1</td>
      <td>5.0</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>396.2</td>
      <td>17.0</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>89.0</td>
      <td>100.0</td>
      <td>27.7</td>
      <td>1.0</td>
      <td>0.9</td>
      <td>8.8</td>
      <td>100.0</td>
      <td>12.0</td>
      <td>24.0</td>
      <td>711.0</td>
      <td>22.0</td>
      <td>396.9</td>
      <td>38.0</td>
      <td>50.0</td>
    </tr>
  </tbody>
</table>
</div>



## Variable Key

| Variable        | Name           
| ------------- |:-------------:|
    | CRIM| per capita crime rate by town  |
| ZN|proportion of residential land zoned for lots over 25,000 sq.ft.      |
| INDUS |proportion of non-retail business acres per town       |
| CHAS |Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)        |
| NOX |nitric oxides concentration (parts per 10 million)       |
| RM |average number of rooms per dwelling |
| AGE |proportion of owner-occupied units built prior to 1940 |
| DIS |weighted distances to five Boston employment centres |
| RAD |index of accessibility to radial highways  |
| TAX |full-value property-tax rate per \$10,000 |
| PTRATIO |pupil-teacher ratio by town  |
| B |1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town  |
| LSTAT |% lower status of the population |
| median_value |Median value of owner-occupied homes in $1000's|

[via UCI](https://archive.ics.uci.edu/ml/datasets/Housing)

## Barebones scatter plot


```python
plot = sb.lmplot(x="RM", y="median_value", data=df)
```


![scatter1](/images/sb_scatter_plot/scatter1.png)

## Add some color and re-label


```python
points = plt.scatter(df["RM"], df["median_value"],
                     c=df["median_value"], s=20, cmap="Spectral") #set style options

#add a color bar
plt.colorbar(points)

#set limits
plt.xlim(3, 9)
plt.ylim(0, 50)

#build the plot
plot = sb.regplot("RM", "median_value", data=df, scatter=False, color=".1")
plot = plot.set(ylabel='Median Home Price ($1000s)', xlabel='Mean Number of Rooms') #add labels
```


![scatter2](/images/sb_scatter_plot/scatter2.png)
