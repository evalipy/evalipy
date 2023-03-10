from evalipy.evalipy import model
import pickle
from sklearn import svm
from sklearn import datasets
from sklearn.linear_model import LinearRegression

line = LinearRegression()


clf = svm.SVC()
X, y = datasets.load_iris(return_X_y=True)
clf.fit(X, y)

s = pickle.dumps(clf)
clf2 = pickle.loads(s)
y_pred = clf2.predict(X[0:1])

model1 = model.Model(model=clf2)
# report = report.Report(model=clf2, actual_data=y, predicted_data=y_pred)
print(model1)
print(model1.raw_type)
print(model.Model(line).raw_type)
# print(report)
