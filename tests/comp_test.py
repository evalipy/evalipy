from evalipy.evalipy import model, report, comparator
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets

tree = DecisionTreeRegressor()
clf = svm.SVC()
X, y = datasets.load_iris(return_X_y=True)
clf.fit(X, y)
tree.fit(X, y)

y_pred_1 = clf.predict(X[0:1])
y_pred_2 = tree.predict(X[0:1])
model_1 = model.Model(model=clf)
model_2 = model.Model(model=tree)

print(model_1)
print(model_2)
print(report.Report(model=model.Model(clf), actual_data=y, predicted_data=y_pred_1))
print(report.Report(model=model.Model(tree), actual_data=y, predicted_data=y_pred_2))

comparator = comparator.Comparator(model_1, model_2, y, y_pred_1, y_pred_2)
print(comparator)
