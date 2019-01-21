import nltk
import numpy as np
import pickle

from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords


from goose3 import Goose
from googlesearch import search

stop_words = set(stopwords.words('english'))

def extract_articles(claim):
	print ('Extracting the articles ...')
	g = Goose()
	articles = []
	for url in search(claim, stop=2):
		if 'snopes' not in url:
			article = g.extract(url=url)
			articles.append(article.cleaned_text)

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
            snippets.append(snippet)
            overlap_score.append(score)
            i += 3

        else:
            i += 1

    return snippets, overlap_score

print ('Enter the claim to check')
claim = input()
articles = extract_articles(claim)

clf = pickle.load(open('stance_classifier.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

print ('Computing the crediblity ...')


per_article_stance = []
evidences = []
for i in range(len(articles)):
	snippets, overlap_score = extract_snippets(claim, articles[i])

	per_snippet_score = []
	if len(snippets) > 0:
		vec = vectorizer.transform(snippets).toarray()
		per_snippet_score.append(clf.predict_proba(vec))
		per_snippet_score = np.array(per_snippet_score).reshape(len(snippets), 2)
		overlap_score = np.array(overlap_score)


		evidences.append(snippets[np.argmax(per_snippet_score[:, 1])])
		for i in range(len(snippets)):
			per_snippet_score[i] = per_snippet_score[i] * overlap_score[i]

		per_article_stance.append(np.argmax(np.sum(per_snippet_score, axis=0)))

if len(per_article_stance) > 0:
	per_article_stance = np.array(per_article_stance)
	count = np.bincount(per_article_stance)
	pred = np.argmax(count)

	print ('Prediction = ', pred)
	for i in range(len(evidences)):
		print ('Evidence ', i, '= ', evidences[i])
		print ('------ X ------')
		print ()

else:
	print ('No articles found')
