import pickle
import numpy as np
import os
import json
import requests
import sys
import nltk
import time


from os.path import join, isfile
from goose3 import Goose, Configuration
from nltk import word_tokenize

from web_res2 import compute_overlap
from vectorize_biaslex import find_vector

from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle	
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler
from sklearn.model_selection import KFold

path = '/Users/aadil/fake_news_detection/Snopes'
dump_path = '/Users/aadil/fake_news_detection/test'

filepaths = [join(path, f) for f in os.listdir(path) if isfile(join(path, f))]
filepaths.sort()


def extract_articles(i):
	articles = []
	g = Goose()


	with open(filepaths[i]) as f:
		urls = []
		data = json.load(f)
		name = data['Claim_ID']
		query = data['Claim']


		for item in data['Google Results'][0]['results']:
			if 'snopes' not in item['link'] and 'pdf' not in item['link']:
				urls.append(item['link'])

		if data['Credibility'] == 'false':
				cred = 0

		else:
			cred = 1

	print ('At claim .... ', name)
	count = 0
	checkpoint = time.time()
	timeout = 5

	for url in urls:
		try:
			## Extracting the articles using goose and extracting the text body and checking overlap with the claim
			article = g.extract(url=url)							
			article = article.cleaned_text
			score, res = compute_overlap(query, article)


			## Checking the limit of maximum 5 articles per claim
			if count > 10:
				break

			## Checking if the overlap is greater than a threshold value
			if res == True:
				articles.append(article)
				count += 1
				checkpoint = time.time()
				filepath = os.path.join(dump_path, name, (str(count) + '.txt'))
				os.makedirs(os.path.dirname(filepath), exist_ok=True)
				f = open(filepath, 'w')
				f.write(article)
				f.close()

			# if time.time() > checkpoint + timeout:
			# 	print ('Timed-out ....', name)
			# 	break

		## Checking for connection error of requests
		except requests.exceptions.ConnectionError as e:
			print ('Some error at file =', name)
			continue

		except:
			# print (sys.exc_info()[0])
			continue

	g.close()
	return articles, cred


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
clf.fit(X, y)

y_test = []
pred = []
for i in range(4300, 4400):
	if i % 20 == 0:
		print ('At step = ', i)

	articles, label = np.array(extract_articles(i))

	if len(articles) > 0:
		X_article =  [[]]

		print ('Recieved something')
		print ()
		for i in range(len(articles)):
			vec = np.array(find_vector(articles[i]))
			X_article.insert(i, vec)

		X_article = np.array(X_article[0:len(articles)])
		X_article = X_article[:, :8]
		scaler.fit_transform(X_article)

		y_pred = clf.predict_proba(X_article)
		pred.append(np.argmax(np.sum(y_pred, axis=0)))
		y_test.append(label)


y_test = np.array(y_test)
pred = np.array(pred)
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
print ('True acc = ', (tp / (tp + fn)))
print ('False acc =', (tn / (fp + tn)))

print ('Pred = ', pred)
print ('Label = ', y_test)

print (pred.shape)
print ('Bincount = ', np.bincount(y_test))

with open('y_test(4300).pkl', 'rb'):
	y_old = pickle.load(f)

y_new = np.concatenate([y_old, y_test])

print ('Dumping y_test')
with open('y_test.pkl', 'wb') as f:
	pickle.dump(y_new, f)

# true_acc = []
# false_acc = []
# for train_index, test_index in kf.split(X):
# 	X_train, y_train = X[train_index], y[train_index]
# 	X_test, y_test = X[test_index], y[test_index]

# 	clf.fit(X_train, y_train)
# 	pred = clf.predict(X_test)

# 	tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
# 	true_acc.append(tp / (tp + fn))
# 	false_acc.append(tn / (fp + tn))

# true_acc = np.array(true_acc)
# false_acc = np.array(false_acc)

# print (clf.predict_proba(X_train))

