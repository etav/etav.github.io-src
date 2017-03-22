---
Title: Selecting Rows And Columns in Python Pandas
Slug: python_pandas_rows_cols
Summary: Selecting Rows And Columns in Python Pandas
Date: 2017-3-22 8:20
Category: Python
Tags: Basics
Authors: Ernest Tavares III
---

Slicing dataframes by rows and columns is a basic tool every analyst should have in their skill-set. We'll run through a quick tutorial covering the basics of selecting rows, columns and both rows and columns.This is an extremely lightweight introduction to rows, columns and pandas—perfect for beginners!

## Import Dataset


```python
import pandas as pd

df = pd.read_csv('iris-data.csv')
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
df.shape
```




    (150, 5)



Selecting the first ten rows


```python
df[:10]
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
    <tr>
      <th>5</th>
      <td>5.4</td>
      <td>3.9</td>
      <td>1.7</td>
      <td>0.4</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4.6</td>
      <td>3.4</td>
      <td>1.4</td>
      <td>0.3</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5.0</td>
      <td>3.4</td>
      <td>1.5</td>
      <td>NaN</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4.4</td>
      <td>2.9</td>
      <td>1.4</td>
      <td>NaN</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>9</th>
      <td>4.9</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>NaN</td>
      <td>Iris-setosa</td>
    </tr>
  </tbody>
</table>
</div>



selecting the last five rows


```python
df[-5:]
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
      <th>145</th>
      <td>6.7</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.3</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <th>146</th>
      <td>6.3</td>
      <td>2.5</td>
      <td>5.0</td>
      <td>2.3</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <th>147</th>
      <td>6.5</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.0</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <th>148</th>
      <td>6.2</td>
      <td>3.4</td>
      <td>5.4</td>
      <td>2.3</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <th>149</th>
      <td>5.9</td>
      <td>3.0</td>
      <td>5.1</td>
      <td>1.8</td>
      <td>Iris-virginica</td>
    </tr>
  </tbody>
</table>
</div>



Selecting rows 15-20


```python
df[15:20]
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
      <th>15</th>
      <td>5.7</td>
      <td>4.4</td>
      <td>1.5</td>
      <td>0.4</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>16</th>
      <td>5.4</td>
      <td>3.9</td>
      <td>1.3</td>
      <td>0.4</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>17</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.3</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>18</th>
      <td>5.7</td>
      <td>3.8</td>
      <td>1.7</td>
      <td>0.3</td>
      <td>Iris-setossa</td>
    </tr>
    <tr>
      <th>19</th>
      <td>5.1</td>
      <td>3.8</td>
      <td>1.5</td>
      <td>0.3</td>
      <td>Iris-setosa</td>
    </tr>
  </tbody>
</table>
</div>



## Selecting Columns

The quickest way to do this using pandas is by providing the column name as the input:


```python
df['class']
```




    0         Iris-setosa
    1         Iris-setosa
    2         Iris-setosa
    3         Iris-setosa
    4         Iris-setosa
    5         Iris-setosa
    6         Iris-setosa
    7         Iris-setosa
    8         Iris-setosa
    9         Iris-setosa
    10        Iris-setosa
    11        Iris-setosa
    12        Iris-setosa
    13        Iris-setosa
    14        Iris-setosa
    15        Iris-setosa
    16        Iris-setosa
    17        Iris-setosa
    18       Iris-setossa
    19        Iris-setosa
    20        Iris-setosa
    21        Iris-setosa
    22        Iris-setosa
    23        Iris-setosa
    24        Iris-setosa
    25        Iris-setosa
    26        Iris-setosa
    27        Iris-setosa
    28        Iris-setosa
    29        Iris-setosa
                ...      
    120    Iris-virginica
    121    Iris-virginica
    122    Iris-virginica
    123    Iris-virginica
    124    Iris-virginica
    125    Iris-virginica
    126    Iris-virginica
    127    Iris-virginica
    128    Iris-virginica
    129    Iris-virginica
    130    Iris-virginica
    131    Iris-virginica
    132    Iris-virginica
    133    Iris-virginica
    134    Iris-virginica
    135    Iris-virginica
    136    Iris-virginica
    137    Iris-virginica
    138    Iris-virginica
    139    Iris-virginica
    140    Iris-virginica
    141    Iris-virginica
    142    Iris-virginica
    143    Iris-virginica
    144    Iris-virginica
    145    Iris-virginica
    146    Iris-virginica
    147    Iris-virginica
    148    Iris-virginica
    149    Iris-virginica
    Name: class, dtype: object




```python
df[['class','petal_width_cm']] #two columns
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class</th>
      <th>petal_width_cm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Iris-setosa</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Iris-setosa</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Iris-setosa</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Iris-setosa</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Iris-setosa</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Iris-setosa</td>
      <td>0.4</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Iris-setosa</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Iris-setosa</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Iris-setosa</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Iris-setosa</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Iris-setosa</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Iris-setosa</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Iris-setosa</td>
      <td>0.1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Iris-setosa</td>
      <td>0.1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Iris-setosa</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Iris-setosa</td>
      <td>0.4</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Iris-setosa</td>
      <td>0.4</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Iris-setosa</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Iris-setossa</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Iris-setosa</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Iris-setosa</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Iris-setosa</td>
      <td>0.4</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Iris-setosa</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Iris-setosa</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Iris-setosa</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Iris-setosa</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Iris-setosa</td>
      <td>0.4</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Iris-setosa</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Iris-setosa</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Iris-setosa</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>120</th>
      <td>Iris-virginica</td>
      <td>2.3</td>
    </tr>
    <tr>
      <th>121</th>
      <td>Iris-virginica</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>122</th>
      <td>Iris-virginica</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>123</th>
      <td>Iris-virginica</td>
      <td>1.8</td>
    </tr>
    <tr>
      <th>124</th>
      <td>Iris-virginica</td>
      <td>2.1</td>
    </tr>
    <tr>
      <th>125</th>
      <td>Iris-virginica</td>
      <td>1.8</td>
    </tr>
    <tr>
      <th>126</th>
      <td>Iris-virginica</td>
      <td>1.8</td>
    </tr>
    <tr>
      <th>127</th>
      <td>Iris-virginica</td>
      <td>1.8</td>
    </tr>
    <tr>
      <th>128</th>
      <td>Iris-virginica</td>
      <td>2.1</td>
    </tr>
    <tr>
      <th>129</th>
      <td>Iris-virginica</td>
      <td>1.6</td>
    </tr>
    <tr>
      <th>130</th>
      <td>Iris-virginica</td>
      <td>1.9</td>
    </tr>
    <tr>
      <th>131</th>
      <td>Iris-virginica</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>132</th>
      <td>Iris-virginica</td>
      <td>2.2</td>
    </tr>
    <tr>
      <th>133</th>
      <td>Iris-virginica</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>134</th>
      <td>Iris-virginica</td>
      <td>1.4</td>
    </tr>
    <tr>
      <th>135</th>
      <td>Iris-virginica</td>
      <td>2.3</td>
    </tr>
    <tr>
      <th>136</th>
      <td>Iris-virginica</td>
      <td>2.4</td>
    </tr>
    <tr>
      <th>137</th>
      <td>Iris-virginica</td>
      <td>1.8</td>
    </tr>
    <tr>
      <th>138</th>
      <td>Iris-virginica</td>
      <td>1.8</td>
    </tr>
    <tr>
      <th>139</th>
      <td>Iris-virginica</td>
      <td>2.1</td>
    </tr>
    <tr>
      <th>140</th>
      <td>Iris-virginica</td>
      <td>2.4</td>
    </tr>
    <tr>
      <th>141</th>
      <td>Iris-virginica</td>
      <td>2.3</td>
    </tr>
    <tr>
      <th>142</th>
      <td>Iris-virginica</td>
      <td>1.9</td>
    </tr>
    <tr>
      <th>143</th>
      <td>Iris-virginica</td>
      <td>2.3</td>
    </tr>
    <tr>
      <th>144</th>
      <td>Iris-virginica</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>145</th>
      <td>Iris-virginica</td>
      <td>2.3</td>
    </tr>
    <tr>
      <th>146</th>
      <td>Iris-virginica</td>
      <td>2.3</td>
    </tr>
    <tr>
      <th>147</th>
      <td>Iris-virginica</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>148</th>
      <td>Iris-virginica</td>
      <td>2.3</td>
    </tr>
    <tr>
      <th>149</th>
      <td>Iris-virginica</td>
      <td>1.8</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 2 columns</p>
</div>



This can be done by providing the column name as a string or by inputing the column number. All three of the methods below produce the same output.

## Selecting Rows & Columns


```python
df['class'][:5] #just first 5 instances
```




    0    Iris-setosa
    1    Iris-setosa
    2    Iris-setosa
    3    Iris-setosa
    4    Iris-setosa
    Name: class, dtype: object




```python
df[df.columns[4]][5:10] #observations 5-10 using 'columns'
```




    5    Iris-setosa
    6    Iris-setosa
    7    Iris-setosa
    8    Iris-setosa
    9    Iris-setosa
    Name: class, dtype: object




```python
df.ix[:, 4][-5:] # last two observations of column using 'ix'
```




    145    Iris-virginica
    146    Iris-virginica
    147    Iris-virginica
    148    Iris-virginica
    149    Iris-virginica
    Name: class, dtype: object
