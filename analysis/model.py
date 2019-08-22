import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from keras.models import Sequential, Model 
from keras.layers import Dense, Embedding, Conv1D, GlobalMaxPool1D, Input, concatenate
from keras import backend as K
from keras import activations
from keras.callbacks import ModelCheckpoint, EarlyStopping
import logging, pickle, random, os
import vis
from . import utils, explain, baselines

if utils.in_ipynb():
	from tqdm import tqdm_notebook as tqdm
	from keras_tqdm import TQDMNotebookCallback
else:
	from tqdm import tqdm

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class CNNModel:

	def __init__(self, \
				 vocab_size, word_index, word2index, emb_dim, emb_matrix, emb_trainable, max_len, target_names, \
				 filters, filter_activations, dense, dense_activations):

		self.vocab_size = vocab_size
		self.word_index = word_index
		self.word2index = word2index
		self.emb_dim = emb_dim
		
		if emb_matrix is not None:
			assert emb_matrix.shape == (vocab_size, emb_dim), "The shape of emb_matrix does not match (vocab_size, emb_dim)"
			self.emb_matrix = emb_matrix
		self.emb_trainable = emb_trainable

		self.max_len = max_len
		self.target_names = target_names

		self.filters = filters
		if filter_activations in ['linear', 'sigmoid', 'relu', 'tanh']:
			self.filter_activations = [filter_activations] * len(filters)
		else:
			self.filter_activations = filter_activations
		assert len(self.filter_activations) == len(filters) and all([ac in ['linear', 'sigmoid', 'relu', 'tanh'] for ac in self.filter_activations]), "Invalid filter_activations" 

		self.dense = dense
		if dense_activations in ['linear', 'sigmoid', 'relu', 'tanh', 'softmax']:
			self.dense_activations = [dense_activations] * len(dense)
		else:
			self.dense_activations = dense_activations
		assert len(self.dense_activations) == len(dense) and all([ac in ['linear', 'sigmoid', 'relu', 'tanh', 'softmax'] for ac in self.dense_activations]), "Invalid dense_activations"

		self.model = self.create_model()
		self.is_Trained = False

	def create_model(self):
		text_input = Input(shape=(None,), dtype='int32')

		if self.emb_matrix is not None:
			embedded_text = Embedding(self.vocab_size, self.emb_dim, 
									   weights=[self.emb_matrix], 
									   input_length=self.max_len, 
									   trainable=self.emb_trainable)(text_input)
		else:
			embedded_text = Embedding(self.vocab_size, self.emb_dim,  
									   input_length=self.max_len)(text_input)

		filters = [Conv1D(f[1], f[0], activation=self.filter_activations[idx])(embedded_text) for idx, f in enumerate(self.filters)]
		max_pools = [GlobalMaxPool1D()(filters[i]) for i in range(len(filters))]
		concatenated = concatenate(max_pools, axis=-1)

		denses = [Dense(self.dense[0], activation=self.dense_activations[0])(concatenated)]
		for idx, d in enumerate(self.dense):
			if idx > 0:
				denses.append(Dense(d, activation=self.dense_activations[idx])(denses[-1]))

		model = Model(text_input, denses[-1])
		model.compile(optimizer='adam',
					  loss='categorical_crossentropy',
					  metrics=['accuracy'])
		model.summary()
		return model

	def train(self, result_foldername, modelname, X_train, y_train_onehot, X_validate, y_validate_onehot, batch_size = 128, epochs = 100, checkpointer = None, early_stopping = None):
		# Create bestmodel_path
		bestmodel_path = result_foldername + '/' + modelname + '.h5'
		if not os.path.exists(result_foldername):
			os.makedirs(result_foldername)
		self.modelname = modelname

		# Define callbacks to use
		if checkpointer is None:
			checkpointer = ModelCheckpoint(filepath=bestmodel_path, verbose=1, save_best_only=True)
		
		if early_stopping is None:
			early_stopping = EarlyStopping(monitor='val_loss', patience=3)

		# Train the model
		utils.__log__("Model training ...")
		if utils.in_ipynb():
			history = self.model.fit(X_train, y_train_onehot, verbose = 0, epochs=epochs, batch_size=batch_size, callbacks=[checkpointer, early_stopping, TQDMNotebookCallback()], validation_data=(X_validate, y_validate_onehot))
		else:
			history = self.model.fit(X_train, y_train_onehot, verbose = 1, epochs=epochs, batch_size=batch_size, callbacks=[checkpointer, early_stopping], validation_data=(X_validate, y_validate_onehot))

		# Load the best weights to the model
		self.model.load_weights(bestmodel_path)
		
		# Memorise training and validation data and prepare components for further analysis
		utils.__log__("Preparing for further analysis ...")
		self.setup_for_analysis(result_foldername, modelname, X_train, y_train_onehot, X_validate, y_validate_onehot)

		utils.__log__("Done")

		return history

	def setup_for_analysis(self, result_foldername, modelname, X_train, y_train_onehot, X_validate, y_validate_onehot, batch_size = 128):
		bestmodel_path = result_foldername + '/' + modelname + '.h5'
		self.modelname = modelname
		self.model.load_weights(bestmodel_path)
		self.X_train, self.y_train_onehot, self.prediction_train = X_train, y_train_onehot, self.predict(X_train, batch_size = batch_size)
		self.X_validate, self.y_validate_onehot, self.prediction_validate = X_validate, y_validate_onehot, self.predict(X_validate, batch_size = batch_size)
		self.get_components()
		self.get_essential_functions()
		self.feature_train_matrix = self.feature_extraction_model.predict(self.X_train, batch_size = batch_size)
		self.feature_validate_matrix = self.feature_extraction_model.predict(self.X_validate, batch_size = batch_size)
		self.class_tendency, self.class_tendency_text = explain.get_class_tendency(self.classification_model, self.feature_train_matrix)
		self.class_identity = np.argmax(self.class_tendency, axis = 1)
		self.threshold_cnnfa = baselines.find_threshold_cnnfa(self.feature_train_matrix, self.prediction_train, self.class_identity)
		self.is_Trained = True
		self.pruned_tree_list = explain.get_pruned_tree_list(self, self.target_names)
		# pickle.dump(self.pruned_tree_list, open(result_foldername + '/' + modelname + '_pruned_tree_list.pickle', 'wb'))
		# explain.draw_tree_list(self.pruned_tree_list, self, folder = result_foldername, prefix = modelname+'_dt')
		

	def get_classification_model(self):
		mlp = Sequential()
		mlp.add(Dense(self.dense[0], activation=self.dense_activations[0], weights = self.model.layers[2*len(self.filters)+3].get_weights(), input_shape = self.model.layers[2*len(self.filters)+3].input_shape[1:]))
		for idx, d in enumerate(self.dense):
			if idx > 0:
				mlp.add(Dense(d, activation=self.dense_activations[idx], weights = self.model.layers[2*len(self.filters)+3+idx].get_weights()))
		mlp.summary()

		mlp.layers[-1].activation = activations.linear
		mlp = vis.utils.utils.apply_modifications(mlp)
		self.classification_model = mlp
		return self.classification_model

	def get_feature_extraction_model(self):
		text_input_fe = Input(shape=(None,), dtype='int32', name='text_indexes')
		embedded_text_fe = Embedding(self.vocab_size, self.emb_dim, 
								   weights=self.model.layers[1].get_weights(), 
								   input_length=self.max_len, 
								   trainable=False)(text_input_fe)

		filters_fe = [Conv1D(f[1], f[0], activation=self.filter_activations[idx], weights = self.model.layers[2 + idx].get_weights())(embedded_text_fe) for idx, f in enumerate(self.filters)]
		max_pools_fe = [GlobalMaxPool1D()(filters_fe[i]) for i in range(len(self.filters))]
		concatenated_fe = concatenate(max_pools_fe, axis=-1)
		fe_model = Model(text_input_fe, concatenated_fe)
		fe_model.summary()
		self.feature_extraction_model = fe_model
		return fe_model

	def get_partial_model(self): # Start from embedded text input
		embedded_text_input = Input(shape=(self.max_len, self.emb_dim), name='embedded_text_input_' + str(random.randint(1,1001)))
		filters_fe = [Conv1D(f[1], f[0], activation=self.filter_activations[idx], weights = self.model.layers[2 + idx].get_weights())(embedded_text_input) for idx, f in enumerate(self.filters)]
		max_pools_fe = [GlobalMaxPool1D()(filters_fe[i]) for i in range(len(self.filters))]
		concatenated_fe = concatenate(max_pools_fe, axis=-1)

		denses = [Dense(self.dense[0], activation=self.dense_activations[0])(concatenated_fe)]
		for idx, d in enumerate(self.dense):
			if idx > 0:
				denses.append(Dense(d, activation=self.dense_activations[idx])(denses[-1]))

		partial_model = Model(embedded_text_input, denses[-1])
		partial_model.summary()
		self.partial_model = partial_model
		return partial_model

	def get_components(self):
		print('Feature extraction model:')
		self.get_feature_extraction_model()

		print('Classification model:')
		self.get_classification_model()

		print('Partial model starting from embedded text matrix:')
		self.get_partial_model()

	def predict(self, X, one_hot = False, batch_size = 32):
		y_pred_onehot = self.model.predict(X, batch_size = batch_size)
		if one_hot:
			return y_pred_onehot
		y_pred = y_pred_onehot.argmax(axis=1).squeeze()
		return y_pred

	def load_weights(self, path):
		self.model.load_weights(path)

	def get_feature_matrix(self, X):
		return self.feature_extraction_model.predict(X, batch_size = 32)

	# def get_convfilters_func(self):
	# 	model = self.model
	# 	return K.function([model.layers[0].input],[model.layers[2 + i].output for i in range(len(self.filters))])

	# def get_embeddings_func(self):
	# 	model = self.model
	# 	return K.function([model.layers[0].input],[model.layers[1].output])

	# def get_features_func(self):
	# 	model = self.model
	# 	return K.function([model.layers[0].input],[model.layers[2*len(self.filters) + 2].output])

	def text2proba(self, textlist):
		if type(textlist) is str:
			textlist = [textlist]
		X = utils.get_data_matrix(textlist, self.word2index, self.max_len)
		return self.predict(X, one_hot = True)

	def get_essential_functions(self):
		model = self.model
		self.convfilters_func = K.function([model.layers[0].input],[model.layers[2 + i].output for i in range(len(self.filters))])
		self.embeddings_func = K.function([model.layers[0].input],[model.layers[1].output])
		self.features_func = K.function([model.layers[0].input],[model.layers[2*len(self.filters) + 2].output])