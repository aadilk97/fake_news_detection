import os
import pickle
import numpy as np

from os.path import join, isfile


path = '/Users/aadil/fake_news_detection/data2'
lex_path = '/Users/aadil/fake_news_detection/public-lexicons/bias_related_lexicons'

filepaths = [join(path, f) for f in os.listdir(path) if isfile(join(path, f))]
filepaths.sort()

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


## Vecotorzing the data using the 2-D list by counting frequency of each feature type.
X = [[]]
y = []
## Ranges to length-1 as the last doc is the test doc(000_test.txt).
for i in range(0, len(filepaths)-1):
	if i % 50 == 0:
		print ('At step ...', i)

	with open(filepaths[i], 'r') as f:
		text = f.read()
		text = text.replace("\"", "")
		text = text.strip()
		tokens = text.split()

		## An empty vector vec for each doc. Updating the vector with frequency of each feature type and then normalizing it.
		vec = []
		for lex in lexicons:
			c = 0
			for word in lex:
				c += tokens.count(word)
			vec.append(c / len(tokens))

		## Inserting the updated vector vec in the array X used for training. Updating the label list y as well.
		X.insert(i, vec)
		y.append(text[-1].rstrip())

## Converting the python lists to numpy arrays.
X = np.array(X[0: len(y)])
y = np.array(y, dtype=int)

## Dumping the vectorized data. Shape: (num_samples, num_features). Each column represents a feature.
with open('X_train_1000.pkl', 'wb') as f:
	pickle.dump(X, f
with open('y_train_1000.pkl', 'wb') as f:
	pickle.dump(y, f)

## Reading the data from the dump
with open('X_train_1000.pkl', 'rb') as f:
	X_train = pickle.load(f)

with open('y_train_1000.pkl', 'rb') as f:
	y_train = pickle.load(f)

print (X_train.shape, y_train.shape)

