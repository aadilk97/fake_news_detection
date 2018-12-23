import nltk
import os
import json
import time
import sys
import requests

from os.path import join, isfile
from goose3 import Goose, Configuration
from nltk import word_tokenize
from time import sleep


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
dump_path = '/Users/aadil/fake_news_detection/data'



filepaths = [join(path, f) for f in os.listdir(path) if isfile(join(path, f))]
filepaths.sort()

start_time = time.time()

c = Configuration()
g = Goose(config=c)


## Do not loop more than 30 items at a time
for i in range(1000, 1100):
	## Closing the old goose instance and starting a new one
	if i % 20 == 0:
		print ('At step .... ', i)
		g.close()
		print ('Restarting goose')
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
			cred = "0\n"

		else:
			print ("True claim = ", name)
			cred = "1\n"

			
	print ('At claim .... ', name)

	articles = []
	count = 0
	checkpoint = time.time()
	timeout = 5

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
				checkpoint = time.time()

			# Check for timeout
			if time.time() > checkpoint + timeout and cred == '0\n':
				print ('Timed-out ....', name)
				break

		## Checking for connection error of requests
		except requests.exceptions.ConnectionError as e:
			print ('Some error at file =', name)
			continue

		except:
			continue


	print ()

print ('Total time = ', ((time.time() - start_time) / 60), 'mins')
	
