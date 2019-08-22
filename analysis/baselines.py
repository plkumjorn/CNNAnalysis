from lime import lime_text
from lime.lime_text import LimeTextExplainer
from .explain import *
from .settings import *
import random
from . import utils, explain
import innvestigate

if utils.in_ipynb():
	from tqdm import tqdm_notebook as tqdm
	from IPython.display import display, HTML
else:
	from tqdm import tqdm

def find_threshold_cnnfa(feature_matrix, prediction_train, class_identity, purity = PURITY_CNNFA):
	threshold_cnnfa = []
	for filter_idx in tqdm(range(feature_matrix.shape[1])):
		X = feature_matrix[:, filter_idx]
		Y = prediction_train
		target_class = class_identity[filter_idx]
		Y_true = (np.array(Y) == target_class)
		DS = list(zip(X,Y))
		DS = sorted(DS, key = lambda x: x[0])
		
		correct = np.sum(Y_true)
		total_example = len(DS)
		for idx, pair in enumerate(DS):
			percentage = correct / (total_example-idx)
			# print(percentage)
			if percentage >= purity:
				threshold_cnnfa.append(pair[0])
				# print(idx, "/", total_example, "-", pair[0])
				break
			if Y_true[idx]:
				correct -= 1
		
		if len(threshold_cnnfa) != filter_idx + 1:
			threshold_cnnfa.append(DS[-1][0])
			# print(idx, "/", total_example, "-", DS[-1][0])
		
		assert len(threshold_cnnfa) == filter_idx + 1
	return threshold_cnnfa

def explain_example_cnnfa(cnn_model, input_text, actual_class = None, is_support = True, print_results = True, print_k = 5):
	threshold_cnnfa, class_of_this_filter, target_names = cnn_model.threshold_cnnfa, cnn_model.class_identity, cnn_model.target_names
	word_index, word2index = cnn_model.word_index, cnn_model.word2index
	fe_input = utils.get_data_matrix([input_text], word2index, cnn_model.max_len, use_tqdm = False)[0]
	conv_features = cnn_model.convfilters_func([np.array([fe_input])])
	features = cnn_model.features_func([np.array([fe_input])])[0]
	predicted_class = cnn_model.predict(np.array([fe_input]))
	processed_text = utils.seq_id2text(word_index, fe_input)
   
	# -------- Find important filters ----------
	selected = features > threshold_cnnfa
	if is_support:
		selected = selected * (class_of_this_filter == predicted_class) # Element-wise AND
	else: # Counter-evidence
		selected = selected * (class_of_this_filter != predicted_class) # Element-wise AND

	positions = [(filter_idx, explain.get_maxngram_position(cnn_model, filter_idx, conv_features), features[0][filter_idx] - threshold_cnnfa[filter_idx]) for filter_idx, use in enumerate(selected[0]) if use]
	positions = sorted(positions, key = lambda x: x[2], reverse = True)
	discriminative_ngrams = [utils.seq_id2text(word_index, fe_input[tuple(pos[1])], pad = True) for pos in positions]
	
	# -------- Find non-overlapping ngrams -----
	included_pos = set()
	non_overlapping_ngrams = []
	for pos in positions:
		if len(set(pos[1][0]).intersection(included_pos)) == 0 and utils.not_all_pad(fe_input[pos[1][0]]): # non-overlap
			included_pos = included_pos.union(set(pos[1][0]))
			non_overlapping_ngrams.append((utils.seq_id2text(word_index, fe_input[tuple(pos[1])], pad = True), list(pos[1][0]), pos[2]))
			if len(non_overlapping_ngrams) >= print_k:
				break
	
	if print_results:
		print("Input text:", input_text)
		print("----------------------------------------------------------------")
		print("Processed text:", processed_text)
		print("----------------------------------------------------------------")
		if actual_class is not None:
			print("Actual class: {} (class id: {})".format(target_names[actual_class], actual_class))
		print("Predicted class: {} (class id: {})".format(target_names[predicted_class], predicted_class))
		print("----------------------------------------------------------------")
		for idx, ngram in enumerate(discriminative_ngrams[:print_k]):
			print("Filter {}: {}".format(positions[idx][0], ngram))
		print("----------------------------------------------------------------")
		exp_type = 'evidence' if is_support else 'counter-evidence'
		print("Non-overlapping ngrams %s:" %(exp_type))
		for idx, ngram in enumerate(non_overlapping_ngrams):
			print("{} (location: {})".format(ngram[0], ngram[1]))
	return non_overlapping_ngrams

def random_ngrams_explanation(cnn_model, input_text, print_results = False, print_k = 5):
	word_index, word2index = cnn_model.word_index, cnn_model.word2index
	fe_input = utils.get_data_matrix([input_text], word2index, cnn_model.max_len, use_tqdm = False)[0]
	
	positions = []
	for i in range(sum([f[1] for f in cnn_model.filters])):
		ngram_len = random.choice([f[0] for f in cnn_model.filters])
		start_position = random.choice(list(range(cnn_model.max_len-ngram_len)))
		positions.append(np.array(list(range(start_position, start_position + ngram_len))))
		
	# -------- Find non-overlapping ngrams -----
	included_pos = set()
	non_overlapping_ngrams = []
	for pos in positions:
		if len(set(pos).intersection(included_pos)) == 0 and utils.not_all_pad(fe_input[pos]): # non-overlap
			included_pos = included_pos.union(set(pos))
			non_overlapping_ngrams.append((utils.seq_id2text(word_index, fe_input[pos], pad = True), list(pos), None))
			if len(non_overlapping_ngrams) >= print_k:
				break

	if print_results:
		print("Non-overlapping ngrams:")
		for idx, ngram in enumerate(non_overlapping_ngrams):
			print("{} (location: {})".format(ngram[0], ngram[1]))

	return non_overlapping_ngrams

def get_scores(keywords, top_ngrams):
	score_1, score_3, score_5 = 0, 0, 0
	for idx, ngram in enumerate(top_ngrams):
		score = sum([keywords.get(w, 0) for w in ngram[0].split()])
		if idx < 1:
			score_1 += score
		if idx < 3:
			score_3 += score
		if idx < 5:
			score_5 += score
	return score_1, score_3, score_5

def compare_LIME_scores(cnn_model, text_test, X_test, y_test, num_test = 100):
	test_ids = list(range(len(X_test)))
	random.shuffle(test_ids)
	num_test = min(num_test, len(X_test))

	score_top_1r, score_top_3r, score_top_5r = [], [], []
	score_top_1o, score_top_3o, score_top_5o = [], [], []
	score_top_1j, score_top_3j, score_top_5j = [], [], []
	score_top_1g, score_top_3g, score_top_5g = [], [], []
	score_top_1h, score_top_3h, score_top_5h = [], [], []

	count = 0
	explainer = LimeTextExplainer(class_names=cnn_model.target_names)
	for example_id in tqdm(test_ids):
		predicted_class = int(cnn_model.predict(np.array([X_test[example_id]])))
		try:
			exp = explainer.explain_instance(text_test[example_id], cnn_model.text2proba, num_features=10, labels=[predicted_class])
		except:
			continue
		keywords = dict([(p[0].lower(), p[1]) for p in list(exp.as_list(label=predicted_class))])
		
		# random method
		top_ngram_random = random_ngrams_explanation(cnn_model, text_test[example_id], print_results = False, print_k = 5)
		# cnnfa et al. method
		top_ngram_cnnfa = explain_example_cnnfa(cnn_model, text_test[example_id], y_test[example_id], print_results = False, print_k = 5)
		# gradient x input method
		top_ngram_ours = explain_prediction(cnn_model, text_test[example_id], y_test[example_id], print_results = False, print_k = 5)
		# gradient (only) method
		top_ngram_grad = explain_prediction(cnn_model, text_test[example_id], y_test[example_id], grad_times_input = False, print_results = False, print_k = 5)
		# gradient x input aggregation method (~Grad-CAM)
		top_ngram_heatmap = explain_prediction_heatmap(cnn_model, text_test[example_id], y_test[example_id], grad_times_input = True, print_results = False, print_k = 5)

		score_1, score_3, score_5 = get_scores(keywords, top_ngram_random)
		score_top_1r.append(score_1); score_top_3r.append(score_3); score_top_5r.append(score_5);
			
		score_1, score_3, score_5 = get_scores(keywords, top_ngram_cnnfa)
		score_top_1j.append(score_1); score_top_3j.append(score_3); score_top_5j.append(score_5);
		
		score_1, score_3, score_5 = get_scores(keywords, top_ngram_ours)
		score_top_1o.append(score_1); score_top_3o.append(score_3); score_top_5o.append(score_5);
		
		score_1, score_3, score_5 = get_scores(keywords, top_ngram_grad)
		score_top_1g.append(score_1); score_top_3g.append(score_3); score_top_5g.append(score_5);
		
		score_1, score_3, score_5 = get_scores(keywords, top_ngram_heatmap)
		score_top_1h.append(score_1); score_top_3h.append(score_3); score_top_5h.append(score_5);
		
		count += 1
		if count == num_test:
			break
		
		print("Random     :", np.mean(score_top_1r), np.mean(score_top_3r), np.mean(score_top_5r))
		print("cnnfa     :", np.mean(score_top_1j), np.mean(score_top_3j), np.mean(score_top_5j))
		print("GradxInput :", np.mean(score_top_1o), np.mean(score_top_3o), np.mean(score_top_5o)) 
		print("Grad       :", np.mean(score_top_1g), np.mean(score_top_3g), np.mean(score_top_5g)) 
		print("Ours       :", np.mean(score_top_1h), np.mean(score_top_3h), np.mean(score_top_5h)) 
		
	print("Random     : %.3f +- %.3f / %.3f +- %.3f / %.3f +- %.3f" % (np.mean(score_top_1r), np.std(score_top_1r), np.mean(score_top_3r), np.std(score_top_3r), np.mean(score_top_5r), np.std(score_top_5r)))  
	print("cnnfa     : %.3f +- %.3f / %.3f +- %.3f / %.3f +- %.3f" % (np.mean(score_top_1j), np.std(score_top_1j), np.mean(score_top_3j), np.std(score_top_3j), np.mean(score_top_5j), np.std(score_top_5j)))
	print("GradxInput : %.3f +- %.3f / %.3f +- %.3f / %.3f +- %.3f" % (np.mean(score_top_1o), np.std(score_top_1o), np.mean(score_top_3o), np.std(score_top_3o), np.mean(score_top_5o), np.std(score_top_5o)))
	print("Grad       : %.3f +- %.3f / %.3f +- %.3f / %.3f +- %.3f" % (np.mean(score_top_1g), np.std(score_top_1g), np.mean(score_top_3g), np.std(score_top_3g), np.mean(score_top_5g), np.std(score_top_5g)))
	print("Ours       : %.3f +- %.3f / %.3f +- %.3f / %.3f +- %.3f" % (np.mean(score_top_1h), np.std(score_top_1h), np.mean(score_top_3h), np.std(score_top_3h), np.mean(score_top_5h), np.std(score_top_5h)))

def explain_example_innvestigate(cnn_model, input_text, method, explain_level = "word", actual_class = None, is_support = True, print_results = True, print_k = 5):
	target_names = cnn_model.target_names
	fe_input = utils.get_data_matrix([input_text], cnn_model.word2index, cnn_model.max_len, use_tqdm = False)[0]
	embedded_matrix = cnn_model.embeddings_func([np.array([fe_input])])[0]
	features = cnn_model.features_func([np.array([fe_input])])[0]
	tokenized_text = [str(w) for w in list(utils.tokenizer(input_text))]
	processed_text = utils.seq_id2text(cnn_model.word_index, fe_input)
	predicted_class = cnn_model.predict(np.array([fe_input]))

	analyzer = innvestigate.create_analyzer(method, innvestigate.utils.model_wo_softmax(cnn_model.partial_model))
	criterion = analyzer.analyze(embedded_matrix)[0] 
	word_level_relevance = np.sum(criterion, axis = 1)[:len(processed_text.split())]
	heatmap = word_level_relevance / np.max(np.abs(word_level_relevance))
	
	if explain_level == "word":
		if is_support:
			non_overlapping_ngrams = [(utils.seq_id2text(cnn_model.word_index, fe_input[[idx]], pad = True), [idx], word_level_relevance[idx]) for idx in np.argsort(-word_level_relevance)[:print_k] if word_level_relevance[idx] > 0]
		else:
			non_overlapping_ngrams = [(utils.seq_id2text(cnn_model.word_index, fe_input[[idx]], pad = True), [idx], -word_level_relevance[idx]) for idx in np.argsort(word_level_relevance)[:print_k] if -word_level_relevance[idx] > 0]
	elif explain_level == "ngram":
		candidate_ngrams = [list(range(start_pos, start_pos + f[0])) for f in cnn_model.filters for start_pos in range(min(len(tokenized_text), cnn_model.max_len)-f[0]+1)]
		candidates = [(ng, sum(np.sum(criterion, axis = 1)[list(ng)])) for ng in candidate_ngrams]
		if is_support:
			candidates = [ng for ng in candidates if ng[1] > 0]
		else:
			candidates = [(ng[0], -ng[1]) for ng in candidates if ng[1] < 0]
		candidates = sorted(candidates, key = lambda x: x[1], reverse = True)
		non_overlapping_ngrams = explain.get_non_overlapping_ngrams(candidates, fe_input, cnn_model.word_index, print_k)

	if print_results:
		print("Input text:", input_text)
		print("----------------------------------------------------------------")
		print("Processed text:", processed_text)
		print("----------------------------------------------------------------")
		if actual_class is not None:
			print("Actual class: {} (class id: {})".format(target_names[actual_class], actual_class))
		print("Predicted class: {} (class id: {})".format(target_names[predicted_class], predicted_class))
		print("----------------------------------------------------------------")
		s = utils.colorize_twoway(processed_text.split(), heatmap)
		display(HTML(s))
		print("----------------------------------------------------------------")
		exp_type = 'evidence' if is_support else 'counter-evidence'
		print("Non-overlapping ngrams %s:" %(exp_type))
		for idx, ngram in enumerate(non_overlapping_ngrams):
			print("{} (location: {})".format(ngram[0], ngram[1]))
	return non_overlapping_ngrams
	