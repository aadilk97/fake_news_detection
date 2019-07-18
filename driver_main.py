import json
import requests
import sys

api_url = 'https://morning-bastion-35625.herokuapp.com/fake_news_api/'

print ('Enter the claim to check')
claim = input()
print ('Computing\n')

url = api_url + claim

d = requests.get(url)
j = json.loads(d.text)

if j['prediction'] == 0:
	print ('The given claim is FALSE.')

else:
	print ('The given claim is TRUE.')

print ('Evidence: ')

evidences = j['evidences']
urls = j['urls']

evidences = evidences.split("\n\n\n")
urls = urls.split('\n')

for i in range(len(evidences)):
	print (evidences[i])
	print ()
	print ('Source: ', urls[i])
	print ('\n' * 3)