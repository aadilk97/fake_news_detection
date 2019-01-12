import os
import pickle
import nltk
import gensim
import numpy as np

from os.path import join, isfile
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer


path = '/Users/aadil/fake_news_detection/review_articles'
lex_path = '/Users/aadil/fake_news_detection/public-lexicons/bias_related_lexicons'
stop_words = set(stopwords.words('english'))



claims = os.listdir(path)

def extract_snippets(claim, article):
	text = ''
	sentences = nltk.sent_tokenize(article)
	sentences = [sent for sent in sentences if len(word_tokenize(sent)) > 3]
	# pp_nouns = [tag[0] for tag in pos_tag(word_tokenize(claim)) if tag[1] in ['NNP', 'NNPS']]

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
			# filtered = [w for w in word_tokenize(snippet) if w not in stop_words]
			# snippet = ' '.join(filtered)
			# snippet = ' '.join([w for w in word_tokenize(snippet) if w not in pp_nouns])
			snippets.append(snippet)
			i += 3

		else:
			i += 1

	return snippets

snippets = []
y = []
for i in range(len(claims)):
	if i % 50 == 0:
		print ('At step ...', i)

	c_path = os.path.join(path, claims[i])

	for article in c_path:
		a_paths = [join(c_path, f) for f in os.listdir(c_path) if isfile(join(c_path, f))]

	claim_query = open(a_paths[-1], 'r')
	claim = claim_query.read()
	index = 0
	for j in range(len(a_paths)-1):
		f = open(a_paths[j], 'r')
		article = f.read()
		article = article.replace("\"", "")
		article = article.strip()	
		label = int(article[-1].rstrip())

		res = extract_snippets(claim, article)
		for snippet in res:
			snippets.append(snippet)
			y.append(label)

# for snippet in snippets:
# 	print (snippet)
# 	print ('------X-------')

# for i in range(len(snippets)):
# 	text = ''
# 	tokens_clean = []
# 	tokens = word_tokenize(snippets[i])
# 	tags = nltk.pos_tag(tokens)
# 	banned = ['NNP', 'NNPS', 'CD']

# 	for tag in tags:
# 		if tag[1] not in banned:
# 			tokens_clean.append(tag[0])

# 	text = ' '.join(tokens_clean)
# 	snippets[i] = text

print ('Loading word2vec model')
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
print ('Model loaded')

X = []
for i in range(len(snippets)):
	vec = []
	allowed = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
	for word in word_tokenize(snippets[i]):
		if word not in stop_words and pos_tag([word])[0][1] in allowed:
			try:
				vec.append(model[word])
			except KeyError:
				continue

	if not vec:
		continue

	vec = np.array(vec)
	final_vec = np.mean(vec, axis=0)
	# min_vec = np.min(vec, axis=0)
	# max_vec = np.max(vec, axis=0)
	# final_vec = np.concatenate((min_vec, max_vec), axis=0)

	X.append(final_vec)

X = np.array(X)

# vectorizer = TfidfVectorizer(min_df=100)
# X = vectorizer.fit_transform(snippets)
y = np.array(y)

print (X.shape, y.shape)

print ('Dumping data')

with open('X_train.pkl', 'wb') as f:
	pickle.dump(X, f)
with open('y_train.pkl', 'wb') as f:
	pickle.dump(y, f)

# with open('vectorizer.pkl', 'wb') as f:
# 	pickle.dump(vectorizer, f)

## Reading the data from the dump
with open('X_train.pkl', 'rb') as f:
	X_train = pickle.load(f)

with open('y_train.pkl', 'rb') as f:
	y_train = pickle.load(f)

print (X_train.shape, y_train.shape)