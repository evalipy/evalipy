# EvaliPy

**EvaliPy** is an evaluation framework for machine learning Models.

The project was started in 2023. It's a package for evaluating different machine learning models and comparing them.  
  
It's currently maintained by me :)

### Dependencies
* Python (>= 3.5)
* NumPy (>= 1.17.3)
* joblib (>= 1.1.1)
* pandas
* matplotlib
* scikit-learn
* scipy
* seaborn

### Installation
```
pip install evalipy
```


### Usage
#### Import
```
from evalipy import *
```
#### Report
```
r = report.Report(model=model.Model(clf), actual_data=y, predicted_data=y_pred_1)
*(optional)* print(r)
```
#### Compare
```
...
tree.fit(X, y)
line.fit(X, y)
...

comparator = comparator.Comparator(models=[linear_model, tree_model], x=X, actual_data=y)
print(comparator)
```


### Authors
* **MR-EIGHT** (Mehrdad Heshmat)
