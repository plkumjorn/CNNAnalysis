# Parts of this block is brought from https://github.com/clips/interpret_with_rules (Sushil et al., 2018)

import sys, math, pickle, spacy, os.path, random, datetime, copy, scipy, csv, json
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter

def load_amazon(train_path, test_path, num_examples = None):
	f = open(train_path, 'r', encoding = 'utf-8')
	tr_dataset = [(line.strip()[11:], int(line.strip()[9])-1) for line in f]
	random.shuffle(tr_dataset)
	
	f = open(test_path, 'r', encoding = 'utf-8')
	te_dataset = [(line.strip()[11:], int(line.strip()[9])-1) for line in f]
	random.shuffle(te_dataset)
	
	train_text, y_train = tuple(zip(*tr_dataset))
	test_text, y_test = tuple(zip(*te_dataset))
	
	if num_examples is None:
		return ['negative', 'positive'], train_text, y_train, test_text, y_test
	else:
		train, val, test = num_examples # such as (100000, 50000, 100000)
		return ['negative', 'positive'], train_text[:train], y_train[:train], train_text[train:train+val], y_train[train:train+val], test_text[:test], y_test[:test]

def load_agnews(agnews_folder, val_ratio = 0.2, shuffle = True): # The dataset can be downloaded from http://goo.gl/JyCnZq
	f = open(agnews_folder + 'classes.txt', 'r')
	target_names = [line.strip() for line in f]

	with open(agnews_folder + 'train.csv', 'r', encoding = 'utf-8') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		tr_dataset = [(row[1] + ' ' + row[2], int(row[0])-1) for row in csv_reader]

	with open(agnews_folder + 'test.csv', 'r', encoding = 'utf-8') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		te_dataset = [(row[1] + ' ' + row[2], int(row[0])-1) for row in csv_reader]

	if shuffle:
		random.shuffle(tr_dataset)
		random.shuffle(te_dataset)

	train_text, y_train = tuple(zip(*tr_dataset))
	test_text, y_test = tuple(zip(*te_dataset))

	val_split = int(len(y_train) * (1- val_ratio))
	return target_names, train_text[:val_split], y_train[:val_split], train_text[val_split:], y_train[val_split:], test_text, y_test

def load_sarcasm_headlines(jsonpath, ratio = [0.6, 0.2, 0.2], shuffle = True):
	dataset = []
	f = open(jsonpath, 'r', encoding = 'utf-8')
	for line in f:
		obj = json.loads(line.strip())
		dataset.append((obj["headline"], obj["is_sarcastic"]))

	if shuffle:
		random.shuffle(dataset)

	train, validation, test = dataset[:int(ratio[0]*len(dataset))], dataset[int(ratio[0]*len(dataset)):int((ratio[0]+ratio[1])*len(dataset))], dataset[int((ratio[0]+ratio[1])*len(dataset)):]
	text_train, y_train = map(list, zip(*train)) 
	text_validate, y_validate = map(list, zip(*validation)) 
	text_test, y_test = map(list, zip(*test)) 
	return ["Not sarcastic", "Sarcastic"], text_train, y_train, text_validate, y_validate, text_test, y_test

def get_train_test_split(x, y, test_ratio = 0.1, seed = 0):
	sss = StratifiedShuffleSplit(n_splits = 1, test_size = test_ratio, random_state = seed)
	for train_idx, val_idx in sss.split(x, y):
		x_train, y_train = [x[i] for i in train_idx.tolist()], y[train_idx]
		x_val, y_val = [x[i] for i in val_idx.tolist()], y[val_idx]

	return x_train, x_val, y_train, y_val

# Parts of this block is brought from https://github.com/clips/interpret_with_rules (Sushil et al., 2018)
def load_20newsgroups(ratio = [0.6, 0.2, 0.2], remove=('headers', 'footers'), categories = ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space']):
	newsgroups = fetch_20newsgroups(subset='all', remove=remove, categories=categories)
	target_names = newsgroups.target_names
	newsgroups = list(zip(newsgroups.data, newsgroups.target))
	random.shuffle(newsgroups)
	train, validation, test = newsgroups[:int(ratio[0]*len(newsgroups))], newsgroups[int(ratio[0]*len(newsgroups)):int((ratio[0]+ratio[1])*len(newsgroups))], newsgroups[int((ratio[0]+ratio[1])*len(newsgroups)):]
	text_train, y_train = map(list, zip(*train)) 
	text_validate, y_validate = map(list, zip(*validation)) 
	text_test, y_test = map(list, zip(*test)) 
	return target_names, text_train, y_train, text_validate, y_validate, text_test, y_test

def understand_data(class_names, y_train, y_test, y_validate = None):
	print("The dataset has {} classes: {}".format(len(class_names), class_names))
	print("Training data has {} examples: {}".format(len(y_train),Counter(y_train)))
	if y_validate is not None:
		print("Validation data has {} examples: {}".format(len(y_validate),Counter(y_validate)))
	print("Testing data has {} examples: {}".format(len(y_test),Counter(y_test)))