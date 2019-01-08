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


path = '/Users/aadil/fake_news_detection/Snopes'
dump_path = '/Users/aadil/fake_news_detection/review_articles'



filepaths = [join(path, f) for f in os.listdir(path) if isfile(join(path, f))]
filepaths.sort()

start_time = time.time() 

c = Configuration()
g = Goose(config=c)

if __name__ == '__main__':

	for i in range(2600, 3000):
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
			
			fact_db = ['snopes.com', 'politifact.com', 'factcheck.org', 'truthorfiction.com']
			
			# for item in data['Google Results'][0]['results']:
			# 	for db in fact_db:
			# 		found = False
			# 		if db in item['link'] and found == False:
			# 			urls.append(item['link'])
			# 			found = True

			for db in fact_db:
				found = False
				for item in data['Google Results'][0]['results']:
					if found == False:
						if db in item['link']:
							urls.append(item['link'])
							found = True


			if data['Credibility'] == 'false':
				cred = "0\n"

			else:
				cred = "1\n"

				
		articles = []
		count = 0
		for url in urls:
			try:
				## Extracting the articles using goose and extracting the text body
				article = g.extract(url=url)							
				article = article.cleaned_text + cred


				filepath = os.path.join(dump_path, name, (str(count) + '.txt'))
				os.makedirs(os.path.dirname(filepath), exist_ok=True)
				f = open(filepath, 'w')
				f.write(article)
				f.close()


				claim_path = os.path.join(dump_path, name, 'claim.txt')
				f = open(claim_path, 'w')
				f.write(query)
				f.close()

				count += 1

			except:
				# print (sys.exc_info()[0])
				continue


	print ('Total time = ', ((time.time() - start_time) / 60), 'mins')
	