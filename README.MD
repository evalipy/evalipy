<h1 align="center">EvaliPy </h1>
<br>


<div style="text-align: center; align=center;">
<a href="https://pypi.org/project/evalipy/" target="_blank">
<img src="https://img.shields.io/badge/pypi%20package-v0.0.5-green">
</a>


<a href="https://snyk.io/test/github/evalipy/evalipy" target="_blank"> 
<img src=https://snyk.io/advisor/python/evalipy/badge.svg>
</a>


<a href="https://pypi.org/project/evalipy/" target="_blank">
<img src=https://static.pepy.tech/personalized-badge/evalipy?period=total&units=international_system&left_color=black&right_color=blue&left_text=Downloads>
</a>
</div>  

<br>


**EvaliPy** is an evaluation framework for machine learning Models.

The project was started in 2023. It's a package for evaluating different machine learning models and comparing them.  
  
It's currently maintained by me :)

## Dependencies
* Python (>= 3.5)
* NumPy (>= 1.17.3)
* joblib (>= 1.1.1)
* pandas (>= 1.5.0)
* scikit-learn (>= 1.2.0)


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
r = Report(model=model.Model(clf), actual_data=y, predicted_data=y_pred_1)
print(r)
display(r.report)
```
### Compare
```python
...
tree_model.fit(X, y)
linear_model.fit(X, y)
...

comparator = Comparator(models=[linear_model, tree_model], x=X, actual_data=y)
print(comparator)
```

## TO DO
* Visualizing reports
* Add more metrics
* Model file health checking
* e.t.c


## Authors
* **MR-EIGHT** (Mehrdad Heshmat)
