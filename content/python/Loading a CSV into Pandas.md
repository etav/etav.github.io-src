---
Title: Importing a CSV Into Pandas
Slug: importing_csv_into_pandas
Summary: Importing a CSV Into Pandas
Date: 2017-1-14 3:20
Category: Python
Tags: Basics
Authors: Ernest Tavares III
---


Import necessary modules


```python
import pandas as pd
import numpy as np
```

Create a toy dataframe (to be converted into csv)


```python
data = {'name':['Ernest', 'Jason', 'Kevin', 'Christine'],
    'job':['Analyst', 'Nerd', 'Teacher', 'Product Manager'],
       'salary':['$60,000','','$70,000','$80,000']}
df = pd.DataFrame(data,columns = ['name','job','salary'])
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>job</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ernest</td>
      <td>Analyst</td>
      <td>$60,000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jason</td>
      <td>Nerd</td>
      <td></td>
    </tr>
    <tr>
      <th>2</th>
      <td>Kevin</td>
      <td>Teacher</td>
      <td>$70,000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Christine</td>
      <td>Product Manager</td>
      <td>$80,000</td>
    </tr>
  </tbody>
</table>
</div>



Export the dataframe to a csv in the current directory


```python
df.to_csv('career_info.csv')
```

Now, load the csv


```python
df = pd.read_csv('career_info.csv')
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>name</th>
      <th>job</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Ernest</td>
      <td>Analyst</td>
      <td>$60,000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Jason</td>
      <td>Nerd</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Kevin</td>
      <td>Teacher</td>
      <td>$70,000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Christine</td>
      <td>Product Manager</td>
      <td>$80,000</td>
    </tr>
  </tbody>
</table>
</div>



Notice Pandas conveniently pulls in the header information without needing specification
