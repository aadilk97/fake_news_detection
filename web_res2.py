import requests
import spacy
import nltk
import os
import json
import time
import eventlet
import urllib

from os.path import join, isfile
from goose3 import Goose, Configuration
from googlesearch import search
from nltk.corpus import stopwords
from nltk import word_tokenize


def compute_overlap(claim, article):
	sentences = nltk.sent_tokenize(article)

	for sent in sentences:
		sent = sent.replace("\"", "")
		tk_claim = word_tokenize(claim)
		tk_sent = word_tokenize(sent)

		c1 = c2 = 0
		for t in tk_claim:
			if t in tk_sent:
				c1 += 1

		bigrm1 = list(nltk.bigrams(tk_claim))
		bigrm2 = list(nltk.bigrams(tk_sent))

		for b in bigrm1:
			if b in bigrm2:
				c2 += 1

		score = c1 + c2
		if score >= 0.4 * (len(tk_claim) + len(bigrm1)):
			return score, True

		
	return 0, False
	print ("----XXX-----")

path = '/Users/aadil/fake_news_detection/Snopes'
dump_path = '/Users/aadil/fake_news_detection/data2'
dump_path2 = '/Users/aadil/fake_news_detection/'



filepaths = [join(path, f) for f in os.listdir(path) if isfile(join(path, f))]
filepaths.sort()


nlp = spacy.load('en')
stop = set(stopwords.words('english'))

for i in range(290, 320):
	if i % 10 == 0:
		print ('At step .... ', i)
	with open(filepaths[i]) as f:
		urls = []
		data = json.load(f)
		name = data['Claim_ID']
		query = data['Claim']

		for item in data['Google Results'][0]['results']:
			if 'snopes' not in item['link'] and 'pdf' not in item['link']:
				urls.append(item['link'])

		if data['Credibility'] == 'false':
			cred = "0\n"

		else:
			print ("True claim = ", name)
			cred = "1\n"

	
	c = Configuration()
	c.http_timeout = 10
	g = Goose(config=c)

	articles = []
	count = 0
	for url in urls:
		try:
			## Extracting the articles using goose and extracting the text body and checking overlap with the claim
			article = g.extract(url=url)							
			article = article.cleaned_text + cred
			score, res = compute_overlap(query, article)

			## Checking the limit of maximum 5 articles per claim
			if count > 5:
				break

			## Checking if the overlap is greater than a threshold value
			if res == True:
				print ('Dumping ....', name)
				f = open(os.path.join(dump_path, (name + str(count) +  '.txt')), 'w')
				f.write(article)
				f.close()

				count += 1

		except:
			continue

	print ("out")
	# query = [i for i in query.lower().split() if i not in stop]
	# print (urls)

	# path = '/Users/aadil/Downloads/bias_related_lexicons/assertives_hooper1975.txt'
	# assertives = []
	# with open(path, 'r') as f:
	# 	for word in f:
	# 		assertives.append(word.rstrip('\n'))

	# # print (news)



	# for j in range(len(articles)):
	# 	score, res = compute_overlap(query, articles[j])
	# 	c = 0
	# 	if res == True and c < 5:
	# 		print ('Dumping ....', name)
	# 		f = open(os.path.join(dump_path, (name + str(j) +  '.txt')), 'w')

	# 		f.write(articles[j])
	# 		f.close()
	# 		# f2.write(cred)
	# 		c += 1


	# 	sentences = nltk.sent_tokenize(news)

	# 	for sent in sentences:
	# 		c = 0
	# 		for word in sent.split():
	# 			if word in query:
	# 				# print ("Word = ", word)
	# 				# print (sent)
	# 				# print ("------xxx------")
	# 				# break
	# 				c += 1

	# 		if c >= len(query) / 2:
	# 			print (sent)
	# 			print ("------xxx------")



