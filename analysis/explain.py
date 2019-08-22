from vis.visualization import visualize_activation, visualize_saliency
from vis.utils import utils
import numpy as np
import pandas as pd
import scipy, copy
from . import utils, baselines, settings
import os, random
os.environ["PATH"] += os.pathsep + settings.GRAPHVIZ_PATH
import graphviz

if utils.in_ipynb():
	from tqdm import tqdm_notebook as tqdm
	from IPython.display import display, HTML
else:
	from tqdm import tqdm
# -------------------------------------------------------------------------------------
# PREDICTION INTERPRETABILITY
# -------------------------------------------------------------------------------------

def get_filter_info(cnn_model, pos_in_features):
	filter_size_count = 0
	for idx, f in enumerate(cnn_model.filters):
		filter_size_count += f[1]
		if pos_in_features < filter_size_count:
			ngram_len = f[0]
			offset = pos_in_features - (filter_size_count - f[1])
			break
	return idx, offset, ngram_len

def convolute(cnn_model, filter_idx, positions, text_matrix):
	filterset_idx, offset, ngram_len = get_filter_info(cnn_model, filter_idx)
	the_input = np.array([tm[positions[idx]] for idx, tm in enumerate(text_matrix)]) 
	the_filter = cnn_model.model.layers[filterset_idx + 2].get_weights()[0][:, :, offset]  
	return np.sum(the_input*the_filter, axis = 2)

def get_maxngram_position(cnn_model, pos_in_features, conv_features):
	# Return a position of n-gram in the text input which corresponds to the value of this feature
	filterset_idx, offset, ngram_len = get_filter_info(cnn_model, pos_in_features)
	conv = conv_features[filterset_idx]
	the_vector = conv[:,:,offset]
	start_positions = np.argmax(the_vector, axis = 1)
	return [np.array(list(range(start_position, start_position + ngram_len))) for start_position in start_positions]

def get_non_overlapping_ngrams(candidates, fe_input, word_index, print_k):
	included_pos = set()
	non_overlapping_ngrams = []
	for pos in candidates:
		if len(set(pos[0]).intersection(included_pos)) == 0 and utils.not_all_pad(fe_input[list(pos[0])]): # non-overlap
			if len(non_overlapping_ngrams) >= print_k or pos[1] <= 0:
				break
			included_pos = included_pos.union(set(pos[0]))
			non_overlapping_ngrams.append((utils.seq_id2text(word_index, fe_input[list(pos[0])], pad = True), list(pos[0]), pos[1]))
			
	return non_overlapping_ngrams

def explain_prediction(cnn_model, input_text, actual_class, grad_times_input = True, print_results = True, print_k = 5):
	word_index, word2index = cnn_model.word_index, cnn_model.word2index
	fe_input = utils.get_data_matrix([input_text], word2index, cnn_model.max_len, use_tqdm = False)[0]
	emb_text = cnn_model.embeddings_func([np.array([fe_input])])[0]
	conv_features = cnn_model.convfilters_func([np.array([fe_input])])
	features = cnn_model.features_func([np.array([fe_input])])[0]
	predicted_class = cnn_model.predict(np.array([fe_input]))
	processed_text = utils.seq_id2text(word_index, fe_input)
   
	# -------- Find important filters ----------
	saliency = visualize_saliency(cnn_model.classification_model, -1, filter_indices = predicted_class, seed_input = features, grad_modifier = 'relu', keepdims = True)
	if grad_times_input:
		criterion = saliency*features[0]
	else:
		criterion = saliency
	argsort_saliency = np.argsort(-criterion)
	positions = [(filter_idx, get_maxngram_position(cnn_model, filter_idx, conv_features), criterion[filter_idx]) for filter_idx in argsort_saliency]
	discriminative_ngrams = [utils.seq_id2text(word_index, fe_input[pos[1]], pad = True) for pos in positions[:print_k]]
	slot_activations = [convolute(cnn_model, pos[0], pos[1], emb_text)[0] for pos in positions[:print_k]]
	
	# -------- Find non-overlapping ngrams -----
	included_pos = set()
	non_overlapping_ngrams = []
	for pos in positions:
		if len(set(pos[1][0]).intersection(included_pos)) == 0 and utils.not_all_pad(fe_input[pos[1][0]]): # non-overlap
			included_pos = included_pos.union(set(pos[1][0]))
			non_overlapping_ngrams.append((utils.seq_id2text(word_index, fe_input[pos[1]], pad = True), pos[1][0]))
			if len(non_overlapping_ngrams) >= print_k:
				break
		
	if print_results:
		print("Input text:", input_text)
		print("----------------------------------------------------------------")
		print("Processed text:", processed_text)
		print("----------------------------------------------------------------")
		print("Actual class: {} (class id: {})".format(cnn_model.target_names[actual_class], actual_class))
		print("Predicted class: {} (class id: {})".format(cnn_model.target_names[predicted_class], predicted_class))
		print("----------------------------------------------------------------")
		print("Discriminative ngrams:")
		for idx, ngram in enumerate(discriminative_ngrams):
			print("Filter {0}: {1} (gi: {2:.3f}, sa: {3}) {4:.3f} {5:.3f}".format(positions[idx][0], ngram, (saliency*features[0])[argsort_saliency[idx]], ', '.join([str(round(score,3)) for score in slot_activations[idx]]), saliency[argsort_saliency[idx]], features[0][argsort_saliency[idx]]))
		print("----------------------------------------------------------------")
		print("Non-overlapping ngrams:")
		for idx, ngram in enumerate(non_overlapping_ngrams):
			print("{} (location: {})".format(ngram[0], ngram[1]))
	return non_overlapping_ngrams

def explain_prediction_heatmap(cnn_model, input_text, actual_class = None, is_support = True, grad_times_input = True, print_results = True, print_k = 5):
	word_index, word2index = cnn_model.word_index, cnn_model.word2index
	fe_input = utils.get_data_matrix([input_text], word2index, cnn_model.max_len, use_tqdm = False)[0]
	emb_text = cnn_model.embeddings_func([np.array([fe_input])])[0]
	conv_features = cnn_model.convfilters_func([np.array([fe_input])])
	features = cnn_model.features_func([np.array([fe_input])])[0]
	predicted_class = cnn_model.predict(np.array([fe_input]))
	processed_text = utils.seq_id2text(word_index, fe_input)
   
	# -------- Find important filters ----------
	if is_support:
		saliency = visualize_saliency(cnn_model.classification_model, -1, filter_indices = predicted_class, seed_input = features, grad_modifier = 'relu', keepdims = True)
	else:
		pos_saliency = visualize_saliency(cnn_model.classification_model, -1, filter_indices = predicted_class, seed_input = features, grad_modifier = 'relu', keepdims = True)
		saliency = visualize_saliency(cnn_model.classification_model, -1, filter_indices = predicted_class, seed_input = features, grad_modifier = 'absolute', keepdims = True)
		saliency = saliency * (pos_saliency <= 0) # Get only the size of negative values	

	if grad_times_input:
		criterion = saliency*features[0]
	else:
		criterion = saliency
	argsort_saliency = np.argsort(-criterion)
	positions = [(filter_idx, get_maxngram_position(cnn_model, filter_idx, conv_features), criterion[filter_idx]) for filter_idx in argsort_saliency]
	
	# -------- Aggregate scores to be heatmap ----------
	heatmap = np.zeros(cnn_model.max_len)
	for triple in positions:
		heatmap[triple[1][0]] += triple[2]
	heatmap = (heatmap - min(heatmap)) / (max(heatmap) - min(heatmap))
	
	# -------- Rank n-grams based on the (average) aggregated scores ----------
	candidate_ngrams = set([tuple(triple[1][0]) for triple in positions])
	candidates = [(ng, sum(heatmap[list(ng)])) for ng in candidate_ngrams]
	candidates = sorted(candidates, key = lambda x: x[1], reverse = True)
	
	# -------- Find non-overlapping ngrams -----
	non_overlapping_ngrams = get_non_overlapping_ngrams(candidates, fe_input, cnn_model.word_index, print_k)
	
	if print_results:
		print("Input text:", input_text)
		print("----------------------------------------------------------------")
		print("Processed text:", processed_text)
		print("----------------------------------------------------------------")
		if actual_class is not None:
			print("Actual class: {} (class id: {})".format(cnn_model.target_names[actual_class], actual_class))
		print("Predicted class: {} (class id: {})".format(cnn_model.target_names[predicted_class], predicted_class))
		print("----------------------------------------------------------------")
		s = utils.colorize(processed_text.split(), heatmap)
		display(HTML(s))
		print("----------------------------------------------------------------")
		exp_type = 'evidence' if is_support else 'counter-evidence'
		print("Non-overlapping ngrams %s:" %(exp_type))
		for idx, ngram in enumerate(non_overlapping_ngrams):
			print("{} (location: {})".format(ngram[0], ngram[1]))
	return non_overlapping_ngrams

# -------------------------------------------------------------------------------------
# MODEL INTERPRETABILITY
# -------------------------------------------------------------------------------------
softmax = lambda x: np.exp(x)/sum(np.exp(x))

def get_class_tendency(mlp, feature_matrix):
	utils.__log__("Calculate Logits")
	logits = mlp.predict(feature_matrix)

	utils.__log__("Find Correlation")
	class_tendency = []

	for filter_idx in tqdm(range(feature_matrix.shape[1])):
		data = feature_matrix[:, filter_idx]
		corr = [scipy.stats.pearsonr(data, logits[:, class_idx])[0] for class_idx in range(logits.shape[1])]
		class_corr_dist = softmax(corr)
		class_tendency.append(class_corr_dist)

	utils.__log__("Done")

	class_tendency_text = ['(' + ','.join([str(round(cix,2)) for cix in ci])+ ')' for ci in class_tendency]
	return class_tendency, class_tendency_text


from sklearn.tree._tree import TREE_LEAF
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn import tree
from .tree_graphviz import *

def get_tree_list(feature_train_matrix, prediction_train, target_names):
	trees = []
	for class_idx, class_name in enumerate(tqdm(target_names)):
		prediction_this_class = (prediction_train == class_idx).astype(int)
		clf = DecisionTreeClassifier()
		clf.fit(feature_train_matrix, prediction_this_class)
		trees.append(clf)
	return trees

def get_pruned_tree_list(cnn_model, target_names):
	if not cnn_model.is_Trained:
		print("Cannot create decision trees as the cnn model has not been trained yet.")
		return 

	print("Creating decision trees ...")
	trees = get_tree_list(cnn_model.feature_train_matrix, cnn_model.prediction_train, target_names)
	
	print("Pruning decision trees ...")
	pruned_trees = []
	for class_idx, t in enumerate(tqdm(trees)):
		# Prune and render pruned trees (prune with validation)
		new_tree = copy.deepcopy(t)
		prune_with_validation(new_tree.tree_, 0, cnn_model.feature_validate_matrix, (cnn_model.prediction_validate == class_idx).astype(int))
		pruned_trees.append(new_tree)

	return pruned_trees

def draw_tree_list(trees, cnn_model, folder, prefix = 'dt', filetype = 'svg'):
	if not cnn_model.is_Trained:
		print("Cannot draw decision trees as the cnn model has not been trained yet.")
		return
	target_names, class_tendency_text = cnn_model.target_names, cnn_model.class_tendency_text
	for class_idx, t in enumerate(tqdm(trees)):
		draw_tree(t, ['Not this class', target_names[class_idx]], class_tendency_text, filetype, folder + '/' + prefix + '_' + cnn_model.modelname + '_' +target_names[class_idx])

def prune_with_validation(inner_tree, index, X_validate, y_validate):
	if inner_tree.children_left[index] == TREE_LEAF: # This node is a leaf
		return True
	else:
		left_indices = X_validate[:, inner_tree.feature[index]] <= inner_tree.threshold[index]
		right_indices = ~(left_indices)
		y_validate_left = [y for idx, y in enumerate(y_validate) if left_indices[idx]]
		y_validate_right = [y for idx, y in enumerate(y_validate) if right_indices[idx]]
		
		left_is_leaf = prune_with_validation(inner_tree, inner_tree.children_left[index], X_validate[left_indices], y_validate_left)
		right_is_leaf = prune_with_validation(inner_tree, inner_tree.children_right[index], X_validate[right_indices], y_validate_right)
		
		if left_is_leaf and right_is_leaf: # both left and right are leaves
			correct_if_merged = np.sum(y_validate == np.argmax(inner_tree.value[index]))
			correct_if_remained = np.sum(y_validate_left == np.argmax(inner_tree.value[inner_tree.children_left[index]])) + np.sum(y_validate_right == np.argmax(inner_tree.value[inner_tree.children_right[index]]))
			if correct_if_merged >= correct_if_remained:
				inner_tree.children_left[index] = TREE_LEAF
				inner_tree.children_right[index] = TREE_LEAF
				return True
			else:
				return False
		else: # cannot merge left and right
			return False   

def predict_from_treelist(treelist, X):
	predict_all = []
	for atree in treelist:
		predict_this_tree = atree.predict_proba(X)[:, 1] # Probability that it outputs this class
		predict_all.append(predict_this_tree)
	predict_all = np.array(predict_all)
	return np.argmax(predict_all, axis = 0)

def draw_tree(t, target_names, class_tendency_text, fileformat, path, show = False):
	dot_data = export_graphviz(t, out_file=None, class_identity = class_tendency_text, class_names=target_names, impurity = None, node_ids=True, filled=True, rounded=True) 
	graph = graphviz.Source(dot_data) 
	graph.format = fileformat
	graph.render(path)  

def tree_stats(inner_tree, index):
	# Return nodes, depths, leaves 
	if inner_tree.children_left[index] == TREE_LEAF: # This node is a leaf
		return 1, 1, 1
	
	else:
		ln, ld, ll = tree_stats(inner_tree, inner_tree.children_left[index])
		rn, rd, rl = tree_stats(inner_tree, inner_tree.children_right[index])
		return ln + rn + 1, max(ld, rd) + 1, ll + rl

def tree_traverse(inner_tree, index, feature_vec, hit_nodes, check_visit = False):
	if inner_tree.children_left[index] == TREE_LEAF: # This node is a leaf
		return hit_nodes
	
	check_var = inner_tree.feature[index]
	threshold = inner_tree.threshold[index]
	
	if feature_vec[check_var] <= threshold:
		if check_visit:
			hit_nodes.append(index) # remember even just visit
		hit_nodes = tree_traverse(inner_tree, inner_tree.children_left[index], feature_vec, hit_nodes, check_visit)
	else:
		hit_nodes.append(index)
		hit_nodes = tree_traverse(inner_tree, inner_tree.children_right[index], feature_vec, hit_nodes, check_visit)
	
	return hit_nodes

def explain_prediction_global(cnn_model, treelist, input_text, actual_class = None, print_results = True):
	word_index, word2index, class_identity, target_names = cnn_model.word_index, cnn_model.word2index, cnn_model.class_identity, cnn_model.target_names
	fe_input = utils.get_data_matrix([input_text], word2index, cnn_model.max_len, use_tqdm = False)[0]
	features = cnn_model.features_func([np.array([fe_input])])[0]
	conv_features = cnn_model.convfilters_func([np.array([fe_input])])
	predicted_class = cnn_model.predict(np.array([fe_input]))
	processed_text = utils.seq_id2text(word_index, fe_input)

	t = treelist[predicted_class]
	hit_nodes = tree_traverse(t.tree_, 0, features[0], [])
	
	hit_ngrams = []
	for node_id in hit_nodes:
		filter_id = t.tree_.feature[node_id]
		class_id = class_identity[filter_id]
		pos = get_maxngram_position(cnn_model, filter_id, conv_features)
		ngram = utils.seq_id2text(word_index, fe_input[tuple(pos)], pad = True)
		hit_ngrams.append((filter_id, class_id, target_names[class_id], ngram, pos))
	hit_ngrams = pd.DataFrame(hit_ngrams, columns = ('Filter ID', 'Class identity', 'Class name', 'N-grams', 'Positions'))
	
	if print_results:
		print("Input text:", input_text)
		print("----------------------------------------------------------------")
		print("Processed text:", processed_text)
		print("----------------------------------------------------------------")
		if actual_class is not None:
			print("Actual class: {} (class id: {})".format(target_names[actual_class], actual_class))
		print("Predicted class: {} (class id: {})".format(target_names[predicted_class], predicted_class))
		print("----------------------------------------------------------------")
		if len(hit_ngrams) == 0:
			print("There is not any support or counter evidence for this input text")
		else:
			support_ngrams = hit_ngrams[hit_ngrams['Class identity'] == predicted_class]
			if len(support_ngrams) > 0:
				print("The following ngrams support the prediction of class {}".format(target_names[predicted_class]))
				print(support_ngrams[['N-grams', 'Positions']])
			
			counter_ngrams = hit_ngrams[hit_ngrams['Class identity'] != predicted_class]
			counter_ngrams = counter_ngrams.sort_values(['Class identity'])
			if len(counter_ngrams) > 0:
				print("The following ngrams oppose the prediction of class {}".format(target_names[predicted_class]))
				print(counter_ngrams[['Class name', 'N-grams', 'Positions']])
				
	return hit_ngrams

def collect_hit_ngrams(t, cnn_model, X_train = None, feature_train_matrix = None):
	X_train = cnn_model.X_train if X_train is None else X_train
	feature_train_matrix = cnn_model.feature_train_matrix if feature_train_matrix is None else feature_train_matrix
	tree_input = feature_train_matrix
	fe_input = X_train
	tree_output = t.predict(tree_input)
	
	# Find examples which hit at each node
	hit_examples_node = dict()
	for idx, feature_vec in enumerate(tqdm(tree_input)):
		hit_nodes = tree_traverse(t.tree_, 0, feature_vec, [])
		for hit_node in hit_nodes:
			hit_examples_node[hit_node] = hit_examples_node.get(hit_node, [])
			hit_examples_node[hit_node].append(idx)
	
	# Find ngrams corresponding to the hits
	hit_ngrams_node = dict()
	for node_id, example_id_list_all in tqdm(hit_examples_node.items()):
		example_id_list = example_id_list_all[:min(len(example_id_list_all), 1000)]
		conv_features = cnn_model.convfilters_func([fe_input[example_id_list,:]])
		position_list = get_maxngram_position(cnn_model, t.tree_.feature[node_id], conv_features)
		slot_activations = convolute(cnn_model, t.tree_.feature[node_id], position_list, cnn_model.embeddings_func([fe_input[example_id_list,:]])[0])
		hit_ngrams_node[node_id] = sorted([(p[0], tree_output[p[0]], np.sum(p[2]), utils.seq_id2text(cnn_model.word_index, fe_input[p[0]][p[1]], pad = True), p[2]) for p in zip(example_id_list, position_list, slot_activations)], key = lambda tu: -tu[2])

	return hit_ngrams_node

def get_representative_ngrams(hit_ngrams_node):
	representative_ngrams_node = dict()
	for node_id, ngram_list in tqdm(hit_ngrams_node.items()):
		ans = dict()
		ans['max'] = ngram_list[0]
		ans['min'] = ngram_list[-1]
		ans['median'] = ngram_list[round((len(ngram_list)-1)/2)]
		ans['mean'] = np.mean(np.array([t[4] for t in ngram_list]), axis = 0) 
		ans['mode'] = scipy.stats.mode([t[3] for t in ngram_list])
		hits = np.array([t[3].split() for t in ngram_list])
		ans['mode_word'] = scipy.stats.mode(hits, axis = 0)
		activations = np.array([t[4] for t in ngram_list])
		ans['max_word'] = [(hits[pos][idx], activations[pos][idx]) for idx, pos in enumerate(np.argmax(activations, axis = 0))]
		representative_ngrams_node[node_id] = ans
	return representative_ngrams_node

# -------------------------------------------------------------------------------------
# UNIFIED METHOD
# -------------------------------------------------------------------------------------

def explain_example(cnn_model, input_text, method, args = {}, exp_type = ['+', '-'], count = 5):
	available_methods = set(['random_words', 'random_ngrams',
							 'cnnfa',
							 'lime',
							 'lrp_words', 'lrp_ngrams',
							 'deeplift_words', 'deeplift_ngrams',
							 'decision_trees', 'grad_cam'
							])
	
	if method not in available_methods:
		assert False, "Unavailable method: %s"%(str(method))

	T2S = {'+': True, '-': False}

	fe_input = utils.get_data_matrix([input_text], cnn_model.word2index, cnn_model.max_len, use_tqdm = False)[0]
	predicted_class = int(cnn_model.predict(np.array([fe_input])))
	tokenized_text = [str(w) for w in list(utils.tokenizer(input_text)) if str(w) != '']
	explanations = {}

	if method == 'random_words':
		random_order_index = list(range(len(tokenized_text)))
		random.shuffle(random_order_index)
		explanations = dict([(etype, [(tokenized_text[w], [w], None) for w in random_order_index[idx*count: idx*count + count]]) for idx, etype in enumerate(exp_type)])
	
	elif method == 'random_ngrams':
		for etype in exp_type:
			explanations[etype] = baselines.random_ngrams_explanation(cnn_model, input_text, print_results = False, print_k = count)

	elif method == 'cnnfa':
		for etype in exp_type:
			explanations[etype] = baselines.explain_example_cnnfa(cnn_model, input_text, is_support = T2S[etype], print_results = False, print_k = count)

	elif method == 'lime':
		explainer = baselines.LimeTextExplainer(class_names=cnn_model.target_names)
		exp = explainer.explain_instance(input_text, cnn_model.text2proba, num_features=4*count, labels=[predicted_class])
		exp_words = exp.as_list(label=predicted_class)
		for etype in exp_type:
			if etype == '+':
				words = [w for w in exp_words if w[1] > 0]
			else:
				words = [w for w in exp_words if w[1] < 0]
			ans = [(w[0], [tokenized_text.index(w[0])], np.abs(w[1])) for w in words if w[0] in tokenized_text]
			explanations[etype] = ans[:min(count, len(ans))]

	elif method == 'lrp_words': 
		for etype in exp_type:
			explanations[etype] = baselines.explain_example_innvestigate(cnn_model, input_text, 'lrp.epsilon', explain_level = "word", is_support = T2S[etype], print_results = False, print_k = count)
			# explanations[etype] = [list(ng[1]) for ng in non_overlapping_ngrams]

	elif method == 'lrp_ngrams':
		for etype in exp_type:
			explanations[etype] = baselines.explain_example_innvestigate(cnn_model, input_text, 'lrp.epsilon', explain_level = "ngram", is_support = T2S[etype], print_results = False, print_k = count)

	elif method == 'deeplift_words': 
		for etype in exp_type:
			explanations[etype] = baselines.explain_example_innvestigate(cnn_model, input_text, "deep_lift.wrapper", explain_level = "word", is_support = T2S[etype], print_results = False, print_k = count)

	elif method == 'deeplift_ngrams':
		for etype in exp_type:
			explanations[etype] = baselines.explain_example_innvestigate(cnn_model, input_text, "deep_lift.wrapper", explain_level = "ngram", is_support = T2S[etype], print_results = False, print_k = count)

	elif method == 'decision_trees': 
		hit_ngrams = explain_prediction_global(cnn_model, cnn_model.pruned_tree_list, input_text, print_results = False)
		for etype in exp_type:
			if etype == '+':
				ngrams = hit_ngrams[hit_ngrams['Class identity'] == predicted_class]
			else:
				ngrams = hit_ngrams[hit_ngrams['Class identity'] != predicted_class]
			ans = [(' '.join([tokenized_text[pos] for pos in list(row['Positions'][0]) if pos < len(tokenized_text)]), list(row['Positions'][0]), None) for index, row in ngrams.iterrows()]
			explanations[etype] = ans[:min(count, len(ans))]

	elif method == 'grad_cam':
		for etype in exp_type:
			explanations[etype] = explain_prediction_heatmap(cnn_model, input_text, is_support = T2S[etype], grad_times_input = True, print_results = False, print_k = count)

	# explanation_texts = {}
	# for etype, alist in explanations.items():
	# 	explanation_texts[etype] = [' '.join([tokenized_text[pos] for pos in sublist if pos < len(tokenized_text)]) for sublist in alist]

	return {'method': method,
			'tokenized_text': tokenized_text,
			'predicted_class': predicted_class,
			'explanations': explanations}