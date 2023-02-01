from evalipy.src.evalipy import model, metrics, report
import pickle
from sklearn import svm
from sklearn import datasets

clf = svm.SVC()
X, y = datasets.load_iris(return_X_y=True)
clf.fit(X, y)

s = pickle.dumps(clf)
clf2 = pickle.loads(s)
y_pred = clf2.predict(X[0:1])

model = model.Model(model=clf2)
report = report.Report(model=clf2, actual_data=y, predicted_data=y_pred)
print(model)
print(report)
