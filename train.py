import pickle
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle	
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler
from sklearn.model_selection import KFold

with open('X.pkl', 'rb') as f:
	X = pickle.load(f)

with open('y.pkl', 'rb') as f:
	y = pickle.load(f)

(X, y) = shuffle(X, y)

X = X[:, :8]
scaler = MinMaxScaler()
scaler.fit_transform(X)

kf = KFold(n_splits=10)
kf.get_n_splits(X)

clf = LogisticRegression(class_weight='balanced', penalty='l1', max_iter=100000, random_state=0, C=10)

true_acc = []
false_acc = []
for train_index, test_index in kf.split(X):
	X_train, y_train = X[train_index], y[train_index]
	X_test, y_test = X[test_index], y[test_index]

	clf.fit(X_train, y_train)
	pred = clf.predict(X_test)

	tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
	true_acc.append(tp / (tp + fn))
	false_acc.append(tn / (fp + tn))

true_acc = np.array(true_acc)
false_acc = np.array(false_acc)

print (np.mean(true_acc), np.mean(false_acc))

