from keras.preprocessing.sequence import pad_sequences
import sys, math, pickle, spacy, os.path, random, datetime, copy, scipy
import numpy as np
import pandas as pd
from collections import Counter
from keras.utils.np_utils import to_categorical

def in_ipynb():
	return type_of_script() == 'jupyter'

def type_of_script():
	try:
		ipy_str = str(type(get_ipython()))
		if 'zmqshell' in ipy_str:
			return 'jupyter'
		if 'terminal' in ipy_str:
			return 'ipython'
	except:
		return 'terminal'
		
if in_ipynb():
	from tqdm import tqdm_notebook as tqdm
else:
	from tqdm import tqdm
		
nlp = spacy.load('en_core_web_sm')
tokenizer = spacy.load('en_core_web_sm', disable = ['tagger', 'parser', 'ner', 'textcat']) # Use only tokenizer
GLOBAL_TIME = datetime.datetime.now()

def __log__(text, find_diff = True):
	global GLOBAL_TIME
	now = datetime.datetime.now()
	if find_diff:
		print(text, 'at', now, '(from last timestamp', now - GLOBAL_TIME,')')
	else:
		print(text, 'at', now)
	GLOBAL_TIME = now

# Glove pre-trained word embeddings can be download at https://nlp.stanford.edu/projects/glove/
def get_emb_dict(emb_path):
	# Thanks to Karishma Malkan (from https://stackoverflow.com/questions/37793118/load-pretrained-glove-vectors-in-python)
	print("Loading Embeddings Model")
	f = open(emb_path,'r', encoding = 'utf-8')
	model = {}
	for line in f:
		splitLine = line.split()
		word = splitLine[0]
		embedding = np.array([float(val) for val in splitLine[1:]])
		model[word] = embedding
	print("Done.",len(model)," words loaded!")
	return model

def get_embedding_matrix(emb_path, max_len, pad_initialisation = "uniform"):
	if emb_path.endswith('.pickle') or emb_path.endswith('.pkl'):
		emb = pickle.load(open(emb_path, "rb"))
	elif emb_path.endswith('.txt'):
		emb = get_emb_dict(emb_path)
	else:
		raise Exception("Unsupported file type for emb")

	word_index = ['PAD','UNK'] + list(emb.keys())
	word2index = dict([(w, idx) for idx, w in enumerate(word_index)])
	mean_vector = np.mean(np.array(list(emb.values())), axis = 0)
	embedding_dim = emb[word_index[2]].shape[0] # Find embedding dimension
	if pad_initialisation == "uniform":
		pad_vector = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
	elif pad_initialisation == "zeros":
		pad_vector = np.zeros(embedding_dim)
	else:
		assert False, "Invalid pad_initialisation"
	embedding_matrix = np.concatenate(([pad_vector, mean_vector], np.array([emb[w] for w in word_index[2:]])), axis = 0)
	vocab_size = len(word_index)	
	return embedding_matrix, vocab_size, embedding_dim, max_len, word_index, word2index

# Get vector of word indices for a document
def get_doc_vector(text, vocab, word2index, maxlen):
	tokens = tokenizer(text)
	vector = []
	for token in tokens:
		if token.text.lower() in vocab:
			vector.append(word2index[token.text.lower()])
		elif token.text.lower().strip() != '':
			vector.append(1) # UNK
		if len(vector) >= maxlen:
			break
	return vector

def get_data_matrix(textlist, word2index, maxlen, use_tqdm = True):
	ans = []
	vocab = set(word2index.keys())
	textlist = tqdm(textlist) if use_tqdm else textlist
	for text in textlist:
		vector = get_doc_vector(text, vocab, word2index, maxlen)
		ans.append(vector)
	ans = np.array(ans)
	ans = pad_sequences(ans, maxlen, dtype='int32', padding='post', truncating='post', value=0.0)
	return ans   

def seq_id2text(word_index, seq, pad = False):
	if pad:
		return ' '.join([word_index[i] if i != 0 else '_' for i in seq])
	else:
		return ' '.join([word_index[i] for i in seq if i != 0])

def not_all_pad(idx_list):
	for i in idx_list:
		if i != 0:
			return True
	return False

def positions2text(tokenized_text, position_list):
	l = len(tokenized_text)
	ans = ''
	ans += ' '.join([tokenized_text[p] for p in position_list if p < l])
	return ans

# Code adapted from https://gist.github.com/ihsgnef/f13c35cd46624c8f458a4d23589ac768
def colorize(words, color_array, max_width_shown = 600):
	# words is a list of words
	# color_array is an array of numbers between 0 and 1 of length equal to words
	template = '<span class="barcode"; style="color: black; background-color: rgba(255, 0, 0, {}); display:inline-block;">{}</span>'
	colored_string = ''
	for word, color in zip(words, color_array):
		colored_string += template.format(color, '&nbsp' + word + '&nbsp')
	return '<div style="width:%dpx">' % max_width_shown + colored_string + '</div>'

def colorize_twoway(words, color_array, max_width_shown = 600):
	# words is a list of words
	# color_array is an array of numbers between 0 and 1 of length equal to words
	template_pos = '<span class="barcode"; style="color: black; background-color: rgba(255, 0, 0, {}); display:inline-block;">{}</span>'
	template_neg = '<span class="barcode"; style="color: black; background-color: rgba(0, 0, 255, {}); display:inline-block;">{}</span>'
	colored_string = ''
	for word, color in zip(words, color_array):
		if color > 0:
			colored_string += template_pos.format(color, '&nbsp' + word + '&nbsp')
		else:
			colored_string += template_neg.format(-color, '&nbsp' + word + '&nbsp')
	return '<div style="width:%dpx">' % max_width_shown + colored_string + '</div>'
