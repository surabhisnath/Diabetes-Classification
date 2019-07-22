from pandas import read_csv
from random import seed
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# setting random seed
seed = 0
random_state = 0


# reading data
data = read_csv("/home/vaibhav/Desktop/classification_scheme1.csv")
data = data.sample(data.shape[0], replace = False, axis = 0, random_state = random_state)
data = data.fillna(0)
print("filled zeroes")

# 80%-20% split
# 5555 is approximately 80% of 6943, the total number of samples
train_file = data.head(5555)
test_file = data.tail(1388)


# creating labels
train_labels = train_file.iloc[:,-1]
test_labels = test_file.iloc[:,-1]


# removing labels from files
unlabelled_train = train_file.iloc[:,:-1]
unlabelled_test = test_file.iloc[:,:-1]
#print("starting grid search")

# applying grid search for finding best parameters
#param_grid = {'C':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],'gamma':[0.0001,0.001, 0.01, 0.1, 0.2, 1,2,3,4,5]}
#clf_svc = GridSearchCV(estimator = SVC(random_state = random_state), param_grid = param_grid, cv = 5, scoring = 'roc_auc')
#clf_svc.fit(unlabelled_train, train_labels)
#print("found best parameters")

# training SVM
#clf = SVC(C = clf_svc.best_params_['C'], gamma = clf_svc.best_params_['gamma'], random_state = random_state)
clf = SVC(C = 0.01, gamma = 1, random_state = random_state)
clf.fit(unlabelled_train, train_labels)
y_pred = clf.predict(unlabelled_test)
print("model training complete")

# finding accuracy
numberCorrect = 0
for i in range(0, len(y_pred)):
    if test_labels.iloc[i] == y_pred[i]:
        numberCorrect += 1
print("Accuracy = ", (numberCorrect/len(y_pred)*(100)), "%")
