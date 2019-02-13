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
from nltk.corpus import stopwords
from googlesearch import search 
from urllib.parse import urlparse



path = '/Users/aadil/fake_news_detection/Snopes'
dump_path = '/Users/aadil/fake_news_detection/test_30hits'
stop_words = set(stopwords.words('english'))


filepaths = [join(path, f) for f in os.listdir(path) if isfile(join(path, f))]
filepaths.sort()


def is_relevant(claim, article):
	sentences = nltk.sent_tokenize(article)
	sentences = [sent for sent in sentences if len(word_tokenize(sent)) > 3]

	i = 0
	snippets = []
	while i < (len(sentences)-3):
		snippet = ''
		for sent in sentences[i:i+3]:
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
			return True

		else:
			i += 1

	return False


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
				item['link'] = item['link'].replace('https', 'http')
				urls.append(item['link'])

		if data['Credibility'] == 'false':
			cred = '0'

		else:
			cred = '1'

		urls_google = []
		try:
			for url in search(query, stop=30):
				if 'snopes' not in url:
					urls_google.append(url)
		except:
			print ('Some error')
			

		for url in urls_google:
			url = url.replace('https', 'http')
			if url not in urls:
				urls.append(url)

	print ('At claim .... ', name)
	count = 0
	checkpoint = time.time()
	timeout = 5
	extracted_articles_domains = []
	for url in urls:
		try:
			## Extracting the articles using goose and extracting the text body and checking overlap with the claim
			article = g.extract(url=url)							
			article = article.cleaned_text
			res = is_relevant(query, article)

			## Checking if the overlap is greater than a threshold value
			if res == True:
				articles.append(article)
				parsed_uri = urlparse(url)
				domain = '{uri.scheme}://{uri.netloc}/'.format(uri=parsed_uri)
				extracted_articles_domains.append(domain)
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
			# print ('Some error at file =', name)
			continue

		except:
			# print (sys.exc_info()[0])
			continue

	g.close()
	return articles, cred, name, query, extracted_articles_domains


for i in range(4080, 4200):
	if i % 20 == 0:
		print ('At step = ', i)

	articles, label, name, claim, domains = np.array(extract_articles(i))

	if len(articles) > 0:
		print ('Recieved something')
		print ()
		label_path = os.path.join(dump_path, name, 'label.txt')
		f = open(label_path, 'w')
		f.write(label)
		f.close()

		claim_path = os.path.join(dump_path, name, 'claim.txt')
		f = open(claim_path, 'w')
		f.write(claim)
		f.close()

		domain_text = ''
		for domain in domains:
			domain_text += domain + '\n'

		domains_path = os.path.join(dump_path, name, 'domains.txt')
		f = open(domains_path, 'w')
		f.write(domain_text)
		f.close()


# with open('y_test(4300).pkl', 'rb'):
# 	y_old = pickle.load(f)

# y_new = np.concatenate([y_old, y_test])

# print ('Dumping y_test')
# with open('y_test.pkl', 'wb') as f:
# 	pickle.dump(y_new, f)

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

