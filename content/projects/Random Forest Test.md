---
Title: Random Forest test
Slug: rf_test
Date: 2016-03-15 10:20
Category: Algorithms
Author: Ernest Tavares III
---

```python
from treeinterpreter import treeinterpreter as ti
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

from sklearn.datasets import load_boston
boston = load_boston()
forest = RandomForestRegressor()

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
df = pd.read_csv(url, sep='\s+', names = ['CRIM', 'ZN', 'INDUS', 'CHAS','NOX','RM', 'AGE', 'DIS','RAD', 'TAX', 'PTRATIO', 'B','LSTAT', 'median_val'])

```

    /Users/ernestt/venv/lib/python2.7/site-packages/matplotlib/font_manager.py:273: UserWarning: Matplotlib is building the font cache using fc-list. This may take a moment.
      warnings.warn('Matplotlib is building the font cache using fc-list. This may take a moment.')



```python
df['median_val'] = df['median_val'] * 1000
df_target = df['median_val']
df_target
df_data = df['median_val']

#plotting housing prices

price = df.groupby('median_val')['median_val'].count()

plt.figure(figsize=(10, 5))
plt.hist(price.values, bins=10, log=False, color = '#F18118')
plt.xlabel('Median Price ($)', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.show()
```


```python

```


```python
#feature importance for boston
forest.fit(boston.data, boston.target)
importance = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
sort = np.argsort(importance)[::-1]

features = []
# Print the feature ranking
print("Feature ranking:")

for f in range(boston.data.shape[1]):
    features.append("%d. feature %d (%f)" % (f + 1, sort[f], importance[sort[f]]))  
    print("%d. feature: %d (%f)" % (f + 1, sort[f], importance[sort[f]]))
```


```python
#df= pd.DataFrame(boston.data, columns = ['CRIM', 'ZN', 'INDUS', 'CHAS','NOX','RM', 'AGE', 'DIS','RAD', 'TAX', 'PTRATIO', 'B','LSTAT'])
#df
```


```python
# Plot the feature importances of the forest
#plt.figure()
#plt.title("Feature importances")
#plt.bar(range(boston.data[:300].shape[1]), importance[sort],
#       color="r", yerr=std[sort], align="center")
#plt.xticks(range(boston.data[:300].shape[1]), sort)
#plt.xlim([-1, boston.data[:300].shape[1]])
#plt.show()
```


```python
#Linear Regression

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(boston.data, boston.target)

e = boston.target #expected
p = model.predict(boston.data) #predicted

print "Linear regression model \n Boston dataset"
print "Mean squared error = %0.3f" % mse(e, p)
print "R2 score = %0.3f" % r2_score(e, p)
```

    Linear regression model
     Boston dataset
    Mean squared error = 21.898
    R2 score = 0.741


    /Users/ernestt/venv/lib/python2.7/site-packages/scipy/linalg/basic.py:884: RuntimeWarning: internal gelsd driver lwork query error, required iwork dimension not returned. This is likely the result of LAPACK bug 0038, fixed in LAPACK 3.2.2 (released July 21, 2010). Falling back to 'gelss' driver.
      warnings.warn(mesg, RuntimeWarning)



```python
#Random Forest
forest.fit(boston.data, boston.target)

e = boston.target
p = forest.predict(boston.data)

print "Random Forest Algorithm on Boston Housing dataset"
print "Mean squared error = %0.2f" % mse(e, p) #mean square error
print "R2 score = %0.2f" % r2_score(e, p) #

#print boston.DESCR
```

    Random Forest Algorithm on Boston Housing dataset
    Mean squared error = 2.85
    R2 score = 0.97



```python
#pair-wise plots
import matplotlib.pyplot as plt

subset = df
axes = pd.tools.plotting.scatter_matrix(df['B'], alpha=.05)
#plt.savefig('scatter_matrix.png')
plt.show()

#plt.figure()
#plt.title("Feature importances")
#plt.bar(range(boston.data[:300].shape[1]), importance[sort],
#       color="r", yerr=std[sort], align="center")

```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    <ipython-input-30-d210314bfca8> in <module>()
          2 import matplotlib.pyplot as plt
          3
    ----> 4 axes = pd.tools.plotting.scatter_matrix(df['B','median_val'], alpha=.05)
          5 #plt.savefig('scatter_matrix.png')
          6 plt.show()


    /Users/ernestt/venv/lib/python2.7/site-packages/pandas/core/frame.pyc in __getitem__(self, key)
       1995             return self._getitem_multilevel(key)
       1996         else:
    -> 1997             return self._getitem_column(key)
       1998
       1999     def _getitem_column(self, key):


    /Users/ernestt/venv/lib/python2.7/site-packages/pandas/core/frame.pyc in _getitem_column(self, key)
       2002         # get column
       2003         if self.columns.is_unique:
    -> 2004             return self._get_item_cache(key)
       2005
       2006         # duplicate columns & possible reduce dimensionality


    /Users/ernestt/venv/lib/python2.7/site-packages/pandas/core/generic.pyc in _get_item_cache(self, item)
       1348         res = cache.get(item)
       1349         if res is None:
    -> 1350             values = self._data.get(item)
       1351             res = self._box_item_values(item, values)
       1352             cache[item] = res


    /Users/ernestt/venv/lib/python2.7/site-packages/pandas/core/internals.pyc in get(self, item, fastpath)
       3288
       3289             if not isnull(item):
    -> 3290                 loc = self.items.get_loc(item)
       3291             else:
       3292                 indexer = np.arange(len(self.items))[isnull(self.items)]


    /Users/ernestt/venv/lib/python2.7/site-packages/pandas/indexes/base.pyc in get_loc(self, key, method, tolerance)
       1945                 return self._engine.get_loc(key)
       1946             except KeyError:
    -> 1947                 return self._engine.get_loc(self._maybe_cast_indexer(key))
       1948
       1949         indexer = self.get_indexer([key], method=method, tolerance=tolerance)


    pandas/index.pyx in pandas.index.IndexEngine.get_loc (pandas/index.c:4154)()


    pandas/index.pyx in pandas.index.IndexEngine.get_loc (pandas/index.c:4018)()


    pandas/hashtable.pyx in pandas.hashtable.PyObjectHashTable.get_item (pandas/hashtable.c:12368)()


    pandas/hashtable.pyx in pandas.hashtable.PyObjectHashTable.get_item (pandas/hashtable.c:12322)()


    KeyError: ('B', 'median_val')



```python

```
