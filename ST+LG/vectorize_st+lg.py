import numpy as np
import os
import nltk
import pickle

from os.path import join, isfile
from nltk import word_tokenize, sent_tokenize, pos_tag
from nltk.corpus import stopwords

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
clf_stance = pickle.load(open('stance_classifier.pkl', 'rb'))

path = '/Users/aadil/fake_news_detection/review_articles_extended'
path2 = '/Users/aadil/fake_news_detection/report_articles'
lex_path = '/Users/aadil/fake_news_detection/public-lexicons/bias_related_lexicons'
stop_words = set(stopwords.words('english'))


def extract_snippets(claim, article):
    text = ''
    sentences = nltk.sent_tokenize(article)
    sentences = [sent for sent in sentences if len(word_tokenize(sent)) > 3]

    # for i in range(len(sentences)):
    #     sentences[i] = ' '.join([w.lower() for w in word_tokenize(sentences[i])])

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

claims = os.listdir(path)
snippets = []
y = []
X = []
for i in range(len(claims)):
	if i % 50 == 0:
		print ('At step ...', i)

	c_path = os.path.join(path, claims[i])
	a_paths = [join(c_path, f) for f in os.listdir(c_path) if isfile(join(c_path, f))]
	claim_query = open(a_paths[-1], 'r')
	claim = claim_query.read()


	c_path2 = os.path.join(path2, claims[i])
	if os.path.isdir(c_path2):
		a_paths = a_paths[0:len(a_paths)-1]
		a_paths2 = [join(c_path2, f) for f in os.listdir(c_path2) if isfile(join(c_path2, f))]
		a_paths = a_paths + a_paths2



	for i in range(len(a_paths)-1):
		f = open(a_paths[i], 'r')
		text = f.read()
		text = text.replace("\"", "")
		text = text.strip()	
		label = int(text[-1].rstrip())
		snippets, overlap_score = extract_snippets(claim, text)

		per_snippet_score = []
		if len(snippets) > 0:
			vec = vectorizer.transform(snippets).toarray()
			per_snippet_score.append(clf_stance.predict_proba(vec))
			per_snippet_score = np.array(per_snippet_score).reshape(len(snippets), 2)
			overlap_score = np.array(overlap_score)
			# for i in range(len(snippets)):
			# 	per_snippet_score[i] = per_snippet_score[i] * overlap_score[i]

			# if len(snippets) > 3:
			# 	per_snippet_score = -np.sort(-per_snippet_score, axis=0)
			# 	per_snippet_score = per_snippet_score[0:3]


			f_l = np.array(find_vector(snippets))
			F = np.concatenate((per_snippet_score, f_l), axis=1)
			for row in F:
				X.append(list(row))

			for k in range(len(snippets)):
				y.append(label)


X, y = np.array(X), np.array(y)
X, y = shuffle(X, y)

clf_st_lg = LogisticRegression(class_weight='balanced', penalty='l2', C=100)
clf_st_lg.fit(X, y)

print ('Dumping classifier ...')
with open('classifier_st-lg.pkl', 'wb') as f:
	pickle.dump(clf_st_lg, f)

with open('y.pkl', 'wb') as f:
	pickle.dump(y, f)

with open('X.pkl', 'wb') as f:
	pickle.dump(X, f)

print ('Complete!')

kf = KFold(n_splits=10)
kf.get_n_splits(X)

true_acc = []
false_acc = []
acc = []
for train_index, test_index in kf.split(X):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]

	clf_st_lg.fit(X_train, y_train)
	pred = clf_st_lg.predict(X_test)

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


