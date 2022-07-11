from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse


from .models import Greeting

import os
import pickle
import nltk
import json
import numpy as np

from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from os.path import join, isfile


from goose3 import Goose
from googlesearch import search

stop_words = set(stopwords.words('english'))


# Create your views here.
def index(request):
    # return HttpResponse('Hello from Python!')
    return render(request, "index.html")

def load_lexicons():
	lex_path = 'hello/lexicons'
	lexpaths = [join(lex_path, f) for f in os.listdir(lex_path) if isfile(join(lex_path, f))]
	lexpaths.sort()
	lexicons = [[]]

	for i in range(len(lexpaths)):
		with open(lexpaths[i], 'r', encoding='latin-1') as f:
			word_list = []
			for word in f:
				word_list.append(word.rstrip())

			lexicons.insert(i, word_list)

	return lexicons

def extract_articles(claim):
	print ('Extracting the articles ...')
	g = Goose()
	articles = []
	urls = []
	for url in search(claim, stop=10):
		if 'snopes' not in url:
			try:
				article = g.extract(url=url)
				articles.append(article.cleaned_text)
				urls.append(url)
			except:
				continue

	return articles, urls

def extract_snippets(claim, article):
	text = ''
	sentences = nltk.sent_tokenize(article)
	sentences = [sent for sent in sentences if len(word_tokenize(sent)) > 3]

	i = 0
	snippets = []
	snippets_cs = []
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
			snippet = ' '.join([w for w in word_tokenize(snippet)])
			snippets_cs.append(snippet)

			snippet = ' '.join([w.lower() for w in word_tokenize(snippet)])
			snippets.append(snippet)

			overlap_score.append(score)
			i += 3

		else:
			i += 1

	return snippets, snippets_cs, overlap_score


def find_vector(corpus, lexicons):
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

def process(request, claim):
	response = {}
	articles, urls = extract_articles(claim)

	clf_st = pickle.load(open('hello/models/stance_classifier.pkl', 'rb'))
	clf_st_lg = pickle.load(open('hello/models/classifier_st-lg.pkl', 'rb'))
	vectorizer = pickle.load(open('hello/models/vectorizer.pkl', 'rb'))

	per_article_pred = []
	evidences = []
	evidence_urls = []
	for i in range(len(articles)):
		snippets, snippets_cs, overlap_score = extract_snippets(claim, articles[i])
		lexicons = load_lexicons()

		per_snippet_score = []
		per_snippet_stance = []
		if len(snippets) > 0:
			vec = vectorizer.transform(snippets).toarray()
			per_snippet_stance.append(clf_st.predict(vec))
			per_snippet_score.append(clf_st.predict_proba(vec))
			per_snippet_score = np.array(per_snippet_score).reshape(len(snippets), 2)
			per_snippet_stance = np.array(per_snippet_stance).reshape(len(snippets), 1)
			overlap_score = np.array(overlap_score)

			evidences.append([snippets_cs[np.argmax(overlap_score)], int(per_snippet_stance[np.argmax(overlap_score)])])
			evidence_urls.append(urls[i])
		
			f_l = np.array(find_vector(snippets, lexicons))
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
	
	pred = int(pred)
	response['prediction'] = pred

	final_evidence = []
	for i in range(len(evidences)):
		if evidences[i][1] == pred:
			final_evidence.append(evidences[i][0])

	response['evidences'] = json.dumps(final_evidence, separators=('\n\n\n', ': '), indent=None)
	response['urls'] = json.dumps(evidence_urls, separators=('\n', ': '), indent=None)
	return JsonResponse(response)


def db(request):

    greeting = Greeting()
    greeting.save()

    greetings = Greeting.objects.all()

    return render(request, "db.html", {"greetings": greetings})
