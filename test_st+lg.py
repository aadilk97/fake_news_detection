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
from sklearn.ensemble import RandomForestClassifier

from vectorize_biaslex import find_vector
from gensim.models.doc2vec import Doc2Vec

path = '/Users/aadil/fake_news_detection/test_extended'
lex_path = '/Users/aadil/fake_news_detection/public-lexicons/bias_related_lexicons'
claims = os.listdir(path)
stop_words = set(stopwords.words('english'))


lexpaths = [join(lex_path, f) for f in os.listdir(lex_path) if isfile(join(lex_path, f))]
lexpaths.sort()

lexicons = [[]]

## Generating a wordlist for each stylistic feature and storing in a 2-D list.
for i in range(len(lexpaths)):
	with open(lexpaths[i], 'r', encoding='latin-1') as f:
		word_list = []
		for word in f:
			word_list.append(word.rstrip())

		lexicons.insert(i, word_list)


def find_vector(corpus):
	X = []
	for text in corpus:
		vec = []
		tokens = word_tokenize(text)
		for lex in lexicons:
			if lex:
				c = 0
				for word in lex:
					c += tokens.count(word)
				vec.append(c / len(tokens))
		X.append(vec)

	return X

def extract_snippets(claim, article):
    text = ''
    sentences = nltk.sent_tokenize(article)
    sentences = [sent for sent in sentences if len(word_tokenize(sent)) > 3]

    i = 0
    snippets = []
    overlap_score = []
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
            snippet = ' '.join([w.lower() for w in word_tokenize(snippet)])
            snippets.append(snippet)
            overlap_score.append(score)
            i += 3

        else:
            i += 1

    return snippets, overlap_score


vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
clf_st = pickle.load(open('stance_classifier.pkl', 'rb'))
clf_st_lg = pickle.load(open('classifier_st-lg.pkl', 'rb'))

## Loading dumped data and training a classifier
# X = pickle.load(open('X.pkl', 'rb'))
# y  = pickle.load(open('y.pkl', 'rb'))
# clf_st_lg = LogisticRegression(class_weight='balanced', penalty='l1', C=10000)
# clf_st_lg.fit(X, y)

# with open('classifier_st-lg.pkl', 'wb') as f:
# 	pickle.dump(clf_st_lg, f)


## Checking the accuracy per claim as per test_data
y_test = []
pred = []
for i in range(len(claims)):
	if i % 10 == 0:
		print ('At step i = ', i)

	if 'soapbox' in claims[i]:
		continue

	c_path = os.path.join(path, claims[i])
	claim_path = os.path.join(c_path, 'claim.txt')


	f = open(claim_path, 'r')
	claim = f.read()
	f.close()

	X_article = [[]]
	for article in c_path:
		a_paths = [join(c_path, f) for f in os.listdir(c_path) if isfile(join(c_path, f))]

	per_article_pred = []
	all_snippets = []
	if len(a_paths) > 0:
		for i in range(len(a_paths)-3):
			f = open(a_paths[i], 'r')
			text = f.read()
			snippets, overlap_score = extract_snippets(claim, text)
			for s in snippets:
				all_snippets.append(s)

			per_snippet_score = []
			if len(snippets) > 0:
				vec = vectorizer.transform(snippets).toarray()
				per_snippet_score.append(clf_st.predict_proba(vec))
				per_snippet_score = np.array(per_snippet_score).reshape(len(snippets), 2)
				overlap_score = np.array(overlap_score)
				# for i in range(len(snippets)):
				# 	per_snippet_score[i] = per_snippet_score[i] * overlap_score[i]

				# if len(snippets) > 3:
				# 	per_snippet_score = -np.sort(-per_snippet_score, axis=0)
				# 	per_snippet_score = per_snippet_score[0:3]

				f_l = np.array(find_vector(snippets))
				F = np.concatenate((per_snippet_score, f_l), axis=1)
				article_prediction = `.predict_proba(F)
				per_article_pred.append((np.average(article_prediction, axis=0)))

		if len(per_article_pred) > 0:
			## Giving the most common class as the overll prediction
			# per_article_pred = np.argmax(per_article_pred, axis=1) 
			# count = np.bincount(per_article_pred)
			# pred.append(np.argmax(count))

			## Calculating the sum of probabilities of each article for overall prediction
			pred.append(np.argmax(np.average(per_article_pred, axis=0)))

			label_path = os.path.join(c_path, 'label.txt')
			f = open(label_path, 'r')
			y_test.append(int(f.read()))
			f.close()

			# if y_test[-1] != pred[-1]:
			# 	print (label_path, 'Actual = ', y_test[-1], 'Pred = ', pred[-1])
			# 	print (np.array(per_article_pred))
				# for s in all_snippets:
				# 	print (s)
				# 	print ('---xx----')
				# 	print ()

y_test = np.array(y_test)
pred = np.array(pred)

print ('Pred = ', pred)
print ('Label = ', y_test)

tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
acc = accuracy_score(y_test, pred)

print ('Acc = ', acc)
print ('True acc = ', (tp / np.bincount(y_test)[1]))
print ('False acc =', (tn / np.bincount(y_test)[0]))

print (pred.shape)
print ('Bincount = ', np.bincount(y_test))
