from pandas import read_csv
from sklearn.model_selection import train_test_split
random_state = 0
data = read_csv('classificationScheme1_allYear_16feat.csv')
data = data.fillna(-1)
data = data.sample(data.shape[0], replace = False, axis = 0, random_state = random_state)

y = data.iloc[:,-1]
X = data.iloc[:,:-1]

from imblearn.over_sampling import RandomOverSampler
from time import time
start = time()
ros = RandomOverSampler(random_state = random_state)
X_resampled, y_resampled = ros.fit_resample(X, y)
print("Took",time()-start,"seconds for over-sampling")
#from collections import Counter

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = random_state)
X_resampled_train, X_resampled_test, y_resampled_train, y_resampled_test = train_test_split(X_resampled, y_resampled, test_size = 0.2, random_state = random_state)

#from sklearn.model_selection import GridSearchCV
#from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
#print('Finding best parameters')
#param_grid = {'C':[1,5,10],'gamma':[0.1,1,10]}
#clf_svc = GridSearchCV(estimator = SVC(random_state = random_state), param_grid = param_grid, cv = 10, scoring = 'roc_auc')
#clf_svc.fit(X_train,y_train)
#print('Building and fitting the model')
#clf = SVC(C = 5, gamma = 1, random_state = random_state)
#clf.fit(X_train,y_train)
#y_pred = clf.predict(X_test)
#print('Predictions done\n\n')
#print(confusion_matrix(y_test, y_pred))
#print("\n\nFinding best parameters")
#param_grid = {'C':[1,5,10,15,20],'gamma':[0.0001,0.001, 0.01, 0.1, 0.2,1,10]}
#clf_svc = GridSearchCV(estimator = SVC(random_state = random_state), param_grid = param_grid, cv = 10, scoring = 'roc_auc')
#clf_svc.fit(X_resampled_train,y_resampled_train)
print("---SVM---")
print('Building and fitting the model')
start = time()
clf = SVC(C = 5, gamma = 1, random_state = random_state)
clf.fit(X_resampled_train,y_resampled_train)
print('Predictions done')
y_pred = clf.predict(X_resampled_test)
print("Took", time()-start,"seconds for model building")
print(confusion_matrix(y_resampled_test, y_pred))
from sklearn.metrics import accuracy_score, matthews_corrcoef
print("Accuracy",accuracy_score(y_resampled_test, y_pred))
print("MCC", matthews_corrcoef(y_resampled_test, y_pred))
from sklearn.metrics import classification_report
print(classification_report(y_resampled_test, y_pred))

print("\n\n---DT---")
from sklearn.tree import DecisionTreeClassifier
print('Building and fitting the model')
start = time()
clf = DecisionTreeClassifier(random_state = random_state)
clf.fit(X_resampled_train,y_resampled_train)
print('Predictions done')
y_pred = clf.predict(X_resampled_test)
print("Took", time()-start,"seconds for model building")
print(confusion_matrix(y_resampled_test, y_pred))
print("Accuracy",accuracy_score(y_resampled_test, y_pred))
print("MCC", matthews_corrcoef(y_resampled_test, y_pred))
print(classification_report(y_resampled_test, y_pred))

print("\n\n---RF---")
from sklearn.ensemble import RandomForestClassifier
print('Building and fitting the model')
start = time()
clf = RandomForestClassifier(n_estimators = 20, random_state = random_state)
clf.fit(X_resampled_train,y_resampled_train)
print('Predictions done')
y_pred = clf.predict(X_resampled_test)
print("Took", time()-start,"seconds for model building")
print(confusion_matrix(y_resampled_test, y_pred))
print("Accuracy",accuracy_score(y_resampled_test, y_pred))
print("MCC", matthews_corrcoef(y_resampled_test, y_pred))
print(classification_report(y_resampled_test, y_pred))

print("\n\n---MLP---")
from sklearn.neural_network import MLPClassifier
print('Building and fitting the model')
start = time()
clf = MLPClassifier(random_state = random_state)
clf.fit(X_resampled_train,y_resampled_train)
print('Predictions done')
y_pred = clf.predict(X_resampled_test)
print("Took", time()-start,"seconds for model building")
print(confusion_matrix(y_resampled_test, y_pred))
print("Accuracy",accuracy_score(y_resampled_test, y_pred))
print("MCC", matthews_corrcoef(y_resampled_test, y_pred))
print(classification_report(y_resampled_test, y_pred))
