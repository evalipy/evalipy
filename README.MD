<h1 align="center">EvaliPy </h1>
<br>

**EvaliPy** is an evaluation framework for machine learning Models.

The project was started in 2023. It's a package for evaluating different machine learning models and comparing them.  
  
It's currently maintained by me :)

## Dependencies
* Python (>= 3.5)
* NumPy (>= 1.17.3)
* joblib (>= 1.1.1)
* pandas (>= 1.5.0)
* matplotlib (>= 3.6.2)
* scikit-learn (>= 1.2.0)
* scipy (>= 1.10.0)
* seaborn (>= 0.12.2)

## Installation
```
pip install evalipy
```


## Usage
### Import
```python
from evalipy import *
```
### Report
```python
r = report.Report(model=model.Model(clf), actual_data=y, predicted_data=y_pred_1)
*(optional)* print(r)
```
### Compare
```python
...
tree_model.fit(X, y)
linear_model.fit(X, y)
...

comparator = comparator.Comparator(models=[linear_model, tree_model], x=X, actual_data=y)
print(comparator)
```


## Authors
* **MR-EIGHT** (Mehrdad Heshmat)
