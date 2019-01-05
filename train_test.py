import os
import pickle
import nltk
import numpy as np
import matplotlib.pyplot as plt

from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from os.path import join, isfile

from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle	
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC

from vectorize_biaslex import find_vector


path = '/Users/aadil/fake_news_detection/test'
claims = os.listdir(path)
stop_words = set(stopwords.words('english'))


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def extract_snippets(claim, article):
    text = ''
    sentences = nltk.sent_tokenize(article)

    i = 0
    snippets = []
    overlap_score = []
    while i < (len(sentences)-4):
        snippet = ''
        for sent in sentences[i:i+4]:
            snippet += sent

        tk_claim = word_tokenize(claim)
        tk_snippet = word_tokenize(snippet)
        tk_clean_snippet = [i for i in tk_snippet if i not in stop_words]

        c1 = c2 = 0
        for t in tk_claim:
            if t in tk_clean_snippet:
                c1 += 1

        bigrm1 = list(nltk.bigrams(tk_claim))
        bigrm2 = list(nltk.bigrams(tk_snippet))

        for b in bigrm1:
            if b in bigrm2:
                c2 += 1

        score = c1 + c2
        if score >= 0.2 * (len(tk_claim) + len(bigrm1)):
            snippets.append(snippet)
            overlap_score.append(score)
            i += 4

        else:
            i += 1

    return snippets, overlap_score

with open('X_train.pkl', 'rb') as f:
	X = pickle.load(f)

with open('y_train.pkl', 'rb') as f:
	y = pickle.load(f)

vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

y = np.array(y)
(X, y) = shuffle(X, y)
# X = X[:, :8]
# X = X[:, :7]
# X = X[:, [0,1,2,3,4,5,6]]



scaler = MinMaxScaler()
# scaler.fit_transform(X)

clf = LogisticRegression(class_weight='balanced', penalty='l1', max_iter=100000, random_state=0, C=100)
clf.fit(X, y)
train_pred = clf.predict(X)

## Training data accuracy
# tn, fp, fn, tp = confusion_matrix(y, train_pred).ravel()
# print ('True acc = ', (tp / (np.bincount(y)[1])))
# print ('False acc =', (tn / (np.bincount(y)[0])))

# cv = ShuffleSplit(n_splits=10, random_state=0)
# # plot_learning_curve(clf, 'Learning curve', X, y, None, cv, n_jobs=4)
# # plt.show()

## Checking the cross validation accuracy per article

kf = KFold(n_splits=10)
kf.get_n_splits(X)

true_acc = []
false_acc = []
acc = []
for train_index, test_index in kf.split(X):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]

	clf.fit(X_train, y_train)
	pred = clf.predict(X_test)

	tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
	true_acc.append(tp / np.bincount(y_test)[1])
	false_acc.append(tn / np.bincount(y_test)[0])
	acc.append(accuracy_score(y_test, pred))

true_acc = np.array(true_acc)
false_acc = np.array(false_acc)
acc = np.array(acc)

print ('Acc = ', np.average(acc))
print ('True Acc = ', np.average(true_acc))
print ('False Acc = ', np.average(false_acc))
print (np.bincount(y))


## Checking the accuracy per claim as per test_data
y_test = []
pred = []
for i in range(len(claims)):
	if i % 10 == 0:
		print ('At step i = ', i)


	c_path = os.path.join(path, claims[i])
	claim_path = os.path.join(c_path, 'claim.txt')


	f = open(claim_path, 'r')
	claim = f.read()
	f.close()

	X_article = [[]]
	for article in c_path:
		a_paths = [join(c_path, f) for f in os.listdir(c_path) if isfile(join(c_path, f))]

	per_article_stance = []
	for i in range(len(a_paths)-2):
		f = open(a_paths[i], 'r')
		text = f.read()
		snippets, overlap_score = extract_snippets(claim, text)

		if len(snippets) > 0:
			vec = vectorizer.transform(snippets)
			per_claim_stance = clf.predict_proba(vec)
			per_article_stance.append(np.argmax(np.sum(per_claim_stance, axis=0)))

	if len(per_article_stance) > 0:
		per_article_stance = np.array(per_article_stance)
		count = np.bincount(per_article_stance)
		pred.append(np.argmax(count))

		label_path = os.path.join(c_path, 'label.txt')
		f = open(label_path, 'r')
		y_test.append(int(f.read()))
		f.close()


y_test = np.array(y_test)
pred = np.array(pred)

print ('Pred = ', pred)
print ('Label = ', y_test)

tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
print ('True acc = ', (tp / np.bincount(y_test)[1]))
print ('False acc =', (tn / np.bincount(y_test)[0]))

print (pred.shape)
print ('Bincount = ', np.bincount(y_test))
