import os
import nltk
import numpy as np
import pickle

from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from os.path import join, isfile

from goose3 import Goose
from googlesearch import search

lex_path = '/Users/aadil/fake_news_detection/public-lexicons/bias_related_lexicons'
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


def extract_articles(claim):
	print ('Extracting the articles ...')
	g = Goose()
	articles = []
	for url in search(claim, stop=30):
		if 'snopes' not in url:
			try:
				article = g.extract(url=url)
				articles.append(article.cleaned_text)
			except:
				continue

	return articles

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

print ('Enter the claim to check')
claim = input()
articles = extract_articles(claim)


clf_st = pickle.load(open('stance_classifier.pkl', 'rb'))
clf_st_lg = pickle.load(open('classifier_st-lg.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

print ('Computing the crediblity ...')


per_article_pred = []
for i in range(len(articles)):
	snippets, overlap_score = extract_snippets(claim, articles[i])

	per_snippet_score = []
	if len(snippets) > 0:
		vec = vectorizer.transform(snippets).toarray()
		per_snippet_score.append(clf_st.predict_proba(vec))
		per_snippet_score = np.array(per_snippet_score).reshape(len(snippets), 2)
		overlap_score = np.array(overlap_score)
		

		f_l = np.array(find_vector(snippets))
		F = np.concatenate((per_snippet_score, f_l), axis=1)
		article_prediction = clf_st_lg.predict_proba(F)
		per_article_pred.append((np.average(article_prediction, axis=0)))

if len(per_article_pred) > 0:
	## Giving the most common class as the overll prediction
	# per_article_pred = np.argmax(per_article_pred, axis=1) 
	# count = np.bincount(per_article_pred)
	# pred = np.argmax(count)

	## Calculating the sum of probabilities of each article for overall prediction
	pred = np.argmax(np.average(per_article_pred, axis=0))
	print (np.array(per_article_pred))
	print ('Prediction = ', pred)
	


