---
Title: Counting and Basic Frequency Plots
Slug: count_basic_freq_plot
Summary: Counting and Basic Frequency Plots
Date: 2017-3-22 8:31
Category: Python
Tags: Basics
Authors: Ernest Tavares III
---

Counting is an essential task required for most analysis projects. The ability to  take counts and visualize them  graphically using frequency plots (histograms) enables the analyst to easily recognize patterns and relationships within the data. Good news is this can be accomplished using python with **just 1 line of code!**


```python
import pandas as pd
%matplotlib inline

df = pd.read_csv('iris-data.csv') #toy dataset
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length_cm</th>
      <th>sepal_width_cm</th>
      <th>petal_length_cm</th>
      <th>petal_width_cm</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['class'][:5]
```




    0    Iris-setosa
    1    Iris-setosa
    2    Iris-setosa
    3    Iris-setosa
    4    Iris-setosa
    Name: class, dtype: object



## Frequency Plot for Categorical Data


```python
df['class'].value_counts() #generate counts
```




    Iris-virginica     50
    Iris-setosa        49
    Iris-versicolor    45
    versicolor          5
    Iris-setossa        1
    Name: class, dtype: int64



Notice that the ```value_counts()``` function automatically provides the classes in decending order. Let's bring it to life with a frequency plot.


```python
df['class'].value_counts().plot()
```





![png](/images/counting_basic_plots/output_6_1.png)


I think a bar graph would be more useful, visually.


```python
df['class'].value_counts().plot('bar')
```




![png](/images/counting_basic_plots/output_8_1.png)



```python
df['class'].value_counts().plot('barh') #horizontal bar plot
```





![png](/images/counting_basic_plots/output_9_1.png)



```python
df['class'].value_counts().plot('barh').invert_yaxis() #horizontal bar plot
```


![png](/images/counting_basic_plots/output_10_0.png)


There you have it, a ranked bar plot for categorical data in just 1 line of code using python!

## Histograms for Numberical Data
You know how to graph categorical data, luckily graphing numerical data is even easier using the ```hist()``` function.


```python
df['sepal_length_cm'].hist() #horizontal bar plot
```






![png](/images/counting_basic_plots/output_13_1.png)





```python
df['sepal_length_cm'].hist(bins = 30) #add granularity
```






![png](/images/counting_basic_plots/output_15_1.png)



```python
df['sepal_length_cm'].hist(bins = 30, range=[4, 8]) #add granularity & range
```





![png](/images/counting_basic_plots/output_16_1.png)



```python
df['sepal_length_cm'].hist(bins = 30, range=[4, 8], facecolor='gray') #add granularity & range & color
```





![png](/images/counting_basic_plots/output_17_1.png)


There you have it, a stylized histogram for numerical data using python in 1 compact line of code.
