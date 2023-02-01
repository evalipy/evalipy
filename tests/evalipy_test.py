from evalipy.evalipy import Model
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets

tree = DecisionTreeRegressor()
clf = svm.SVC()
X, y = datasets.load_iris(return_X_y=True)
clf.fit(X, y)

m = Model(clf)
print(m)
