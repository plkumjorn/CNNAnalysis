
# coding: utf-8

# # Amazon Datasets

import sys
sys.path.append("..")
from analysis import *
from pprint import pprint as pprint
import json, csv

# ## Download embeddings
# - Specify max_len at this step

emb_matrix, vocab_size, emb_dim, max_len, word_index, word2index = get_embedding_matrix(EMBEDDINGS_PATH['glove-200'], max_len = 150)

# Project_setup
result_folder = '../results/'
project_name = 'Amazon'
project_path = result_folder + project_name
model_name_fulltime = 'Amazon_general'
model_name_rush = 'Amazon_rush'

# ## Download the datasets

folder_path = "../data/amazonreviews/"
METHODS = ['random_words', 'random_ngrams', 'lime', 'lrp_words', 'lrp_ngrams', 'deeplift_words', 'deeplift_ngrams','decision_trees', 'grad_cam']

y_train = pickle.load(open(folder_path + "y_train_100000.pickle", "rb"))
y_validate = pickle.load(open(folder_path + "y_validate_50000.pickle", "rb"))
y_test = pickle.load(open(folder_path + "y_test_100000.pickle", "rb"))

text_train = pickle.load(open(folder_path + "text_train_100000.pickle", "rb"))
text_validate = pickle.load(open(folder_path + "text_validate_50000.pickle", "rb"))
text_test = pickle.load(open(folder_path + "text_test_100000.pickle", "rb"))

target_names = ['negative', 'positive']

X_train = get_data_matrix(text_train, word2index, maxlen)
X_validate = get_data_matrix(text_validate, word2index, maxlen)
X_test = get_data_matrix(text_test, word2index, maxlen)

y_train_onehot, y_validate_onehot, y_test_onehot = to_categorical(y_train), to_categorical(y_validate), to_categorical(y_test) 
understand_data(target_names, y_train, y_test, y_validate)


# ## Model creation and training

# ### Model 1: Train until the validation loss does not improve for three epochs

cnn_model = CNNModel(vocab_size, word_index, word2index, emb_dim, emb_matrix, False, max_len, target_names,                      filters = [(2, 50), (3, 50), (4, 50)],                      filter_activations = 'relu',                      dense = [150, len(target_names)],                      dense_activations = ['relu', 'softmax'])

cnn_model.train(project_path, model_name_fulltime, X_train, y_train_onehot, X_validate, y_validate_onehot, batch_size = 2048)


# # ### Model 2: Train for only one epoch

cnn_model_rush = CNNModel(vocab_size, word_index, word2index, emb_dim, emb_matrix, False, max_len, target_names,                      filters = [(2, 50), (3, 50), (4, 50)],                      filter_activations = 'relu',                      dense = [150, len(target_names)],                      dense_activations = ['relu', 'softmax'])

cnn_model_rush.train(project_path, model_name_rush, X_train, y_train_onehot, X_validate, y_validate_onehot, batch_size = 2048, epochs = 1)


# # ## Evaluating both models

prediction_test_onehot = cnn_model.predict(X_test, batch_size = 128, one_hot = True)
prediction_test = prediction_test_onehot.argmax(axis=1).squeeze()
print(classification_report(y_test, prediction_test, target_names=target_names))

feature_test_matrix = cnn_model.feature_extraction_model.predict(X_test)
prediction_test_byprunedtreelist = predict_from_treelist(cnn_model.pruned_tree_list, feature_test_matrix)
print("Fidelity: ")
print(classification_report(prediction_test, prediction_test_byprunedtreelist, target_names=target_names))
print("Accuracy: ")
print(classification_report(y_test, prediction_test_byprunedtreelist, target_names=target_names))
print("Tree stats:")
for idx, t in enumerate(tqdm(cnn_model.pruned_tree_list)):
    n, d, l = tree_stats(t.tree_, 0)
    print("{}: Node {}, Depth {}, Leaves {}".format(target_names[idx], n, d, l))
pickle.dump(cnn_model.pruned_tree_list, open(project_path + '/' + model_name_fulltime + '_pruned_tree_list.pickle', 'wb'))
draw_tree_list(cnn_model.pruned_tree_list, cnn_model, folder = project_path)

print("-----------------------------------------------------------------")

prediction_test_onehot_rush = cnn_model_rush.predict(X_test, batch_size = 128, one_hot = True)
prediction_test_rush = prediction_test_onehot_rush.argmax(axis=1).squeeze()
print(classification_report(y_test, prediction_test_rush, target_names=target_names))

feature_test_matrix_rush = cnn_model_rush.feature_extraction_model.predict(X_test)
prediction_test_byprunedtreelist_rush = predict_from_treelist(cnn_model_rush.pruned_tree_list, feature_test_matrix_rush)
print("Fidelity: ")
print(classification_report(prediction_test_rush, prediction_test_byprunedtreelist_rush, target_names=target_names))
print("Accuracy: ")
print(classification_report(y_test, prediction_test_byprunedtreelist_rush, target_names=target_names))
print("Tree stats:")
for idx, t in enumerate(tqdm(cnn_model_rush.pruned_tree_list)):
    n, d, l = tree_stats(t.tree_, 0)
    print("{}: Node {}, Depth {}, Leaves {}".format(target_names[idx], n, d, l))
pickle.dump(cnn_model_rush.pruned_tree_list, open(project_path + '/' + rush + '_pruned_tree_list.pickle', 'wb'))
draw_tree_list(cnn_model_rush.pruned_tree_list, cnn_model_rush, folder = project_path)


# ## User Study A: Trustworthiness

# - Select 100 test texts for testing
#     - 50 texts : both CNNs correctly classify (positive: 25, negative: 25)
#     - 50 texts : both CNNs misclassify (positive: 25, negative: 25)


TP, TN, FP, FN = set([]), set([]), set([]), set([])
for example_idx, y in enumerate(tqdm(y_test)):
    if prediction_test[example_idx] == prediction_test_rush[example_idx]:
        if y == prediction_test[example_idx]: # Correct classify
            if y == 1:
                TP.add(example_idx)
            else:
                TN.add(example_idx)
        else: # Misclassify
            if y == 1:
                FP.add(example_idx) # Wrong notation --> This should be FN
            else:
                FN.add(example_idx)
print([len(aset) for aset in [TP, TN, FP, FN]])
TP, TN, FP, FN = list(TP)[:25], list(TN)[:25], list(FP)[:25], list(FN)[:25]


# - Write tokenized texts into a file

with open(project_path + '/user_study_a_tokenizedtexts.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file, lineterminator="\n")
    writer.writerow(['example_id', 'text', 'tokenized_text', 'actual_class', 'predicted_class'])
    for alist in [TP, TN, FP, FN]:
        for example_id in alist:
            tokenized_text = list(tokenizer(text_test[example_id]))
            tokenized_text = [str(w) for w in tokenized_text]
            writer.writerow([example_id, text_test[example_id], json.dumps(tokenized_text), y_test[example_id], prediction_test[example_id]])


# - Dump explanations to a file

# CNN Model full time training
with open(project_path + '/user_study_a_explanations_full.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file, lineterminator="\n")
    writer.writerow(['example_id', 'explain_method', 'ngrams'])
    for alist in to_use:
        for idx, example_id in enumerate(tqdm(alist)):
            print(idx '; Example_id', example_id)
            for method in METHODS:
                ans = explain_example(cnn_model, text_test[example_id], method, exp_type = ['+'])
                exps = [list(t[1]) for t in ans['explanations']['+']] + [list([]) for i in range(5-len(ans['explanations']['+']))]
                exps = [[int(idx) for idx in ng] for ng in exps]
                writer.writerow([example_id, method, json.dumps(exps)])
                utils.__log__(method)
            print("-----------------------------------")

# In[31]:


# CNN Model rush time training
with open(project_path + '/user_study_a_explanations_rush.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file, lineterminator="\n")
    writer.writerow(['example_id', 'explain_method', 'ngrams'])
    for alist in to_use:
        for example_id in tqdm(alist):
            for method in METHODS:
                ans = explain_example(cnn_model_rush, text_test[example_id], method, exp_type = ['+'])
                exps = [list(t[1]) for t in ans['explanations']['+']] + [list([]) for i in range(5-len(ans['explanations']['+']))]
                exps = [[int(idx) for idx in ng] for ng in exps]
                writer.writerow([example_id, method, json.dumps(exps)])
                utils.__log__(method)
            print("-----------------------------------")


# # ## User Study B: Class Discriminative

# # - Using only the good CNN, select 100 test texts for testing
# #     - 50 texts : The CNN correctly classifies with 90% confidence or more (positive: 25, negative: 25)
# #     - 50 texts : The CNN misclassifies with 90% confidence or more (positive: 25, negative: 25)

tp, tn, fp, fn = set([]), set([]), set([]), set([])
is_correct = prediction_test == y_test
max_confidence = np.max(prediction_test_onehot, axis = 1)
for example_idx, y in enumerate(tqdm(y_test)):
    if max_confidence[example_idx] > 0.9:
        if is_correct[example_idx]: # Correct classify
            if y == 1:
                tp.add(example_idx)
            else:
                tn.add(example_idx)
        else: # Misclassify
            if y == 1:
                fn.add(example_idx)
            else:
                fp.add(example_idx)
print([len(aset) for aset in [tp, tn, fp, fn]])
tp, tn, fp, fn = list(tp)[:25], list(tn)[:25], list(fp)[:25], list(fn)[:25]


# # - Write texts into a file

with open(project_path + '/user_study_b_texts.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file, lineterminator="\n")
    writer.writerow(['example_id', 'text', 'actual_class', 'predicted_class'])
    for alist in [tp, tn, fp, fn]:
        for example_id in alist:
            writer.writerow([example_id, text_test[example_id], y_test[example_id], prediction_test[example_id]])


# # - Dump explanations to a file

# # CNN Model full time training
with open(project_path + '/user_study_b_explanations_full.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file, lineterminator="\n")
    writer.writerow(['example_id', 'explain_method', 'ngrams'])
    for alist in [tp, tn, fp, fn]:
	    for idx, example_id in enumerate(tqdm(alist)):
	        tokenized_text = [str(w) for w in list(utils.tokenizer(input_text)) if str(w) != '']
			for method in METHODS:
	            ans = explain_example(cnn_model, text_test[example_id], method, exp_type = ['+'])
	            exps = [positions2text(tokenized_text , t[1]) for t in ans['explanations']['+']] + ['' for i in range(5-len(ans['explanations']['+']))]
	            writer.writerow([example_id, method, json.dumps(exps)])
	            utils.__log__(method)
	        print("-----------------------------------")

# # ## User Study C: Providing Information for Decision Making

# # - Using only the good CNN, select 100 test texts for testing
# #     - 50 texts : The CNN correctly classifies with 70% confidence or less (positive: 25, negative: 25)
# #     - 50 texts : The CNN misclassifies with 70% confidence or less (positive: 25, negative: 25)

tpp, tnn, fpp, fnn = set([]), set([]), set([]), set([])
is_correct = prediction_test == y_test
max_confidence = np.max(prediction_test_onehot, axis = 1)
for example_idx, y in enumerate(tqdm(y_test)):
    if max_confidence[example_idx] < 0.7:
        if is_correct[example_idx]: # Correct classify
            if y == 1:
                tpp.add(example_idx)
            else:
                tnn.add(example_idx)
        else: # Misclassify
            if y == 1:
                fnn.add(example_idx)
            else:
                fpp.add(example_idx)
print([len(aset) for aset in [tpp, tnn, fpp, fnn]])
tpp, tnn, fpp, fnn = list(tpp)[:25], list(tnn)[:25], list(fpp)[:25], list(fnn)[:25]


# # # - Write texts into a file

with open(project_path + '/user_study_c_texts_with_scores.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file, lineterminator="\n")
    writer.writerow(['example_id', 'text', 'actual_class', 'predicted_class', 'scores'])
    for alist in [tpp, tnn, fpp, fnn]:
        for example_id in alist:
	        scores = list(prediction_test_onehot[example_id])
	        scores = [round(float(s), 3) for s in scores]
	        writer.writerow([example_id, text_test[example_id], y_test[example_id], prediction_test[example_id], json.dumps(scores)])

# # - Dump explanations to a file

# # CNN Model full time training
with open(project_path + '/user_study_c_explanations_full.csv', mode='a') as csv_file:
    writer = csv.writer(csv_file, lineterminator="\n")
    writer.writerow(['example_id', 'explain_method', 'ngrams_support', 'ngrams_oppose'])
    for alist in [tpp, tnn, fpp, fnn]:
	    for idx, example_id in enumerate(tqdm(alist)):
	        print(idx, '; Example_id', example_id)
	        tokenized_text = [str(w) for w in list(utils.tokenizer(input_text)) if str(w) != '']
			for method in METHODS:
	            ans = explain_example(cnn_model, text_test[example_id], method, exp_type = ['+', '-'])
	            exps_support = [positions2text(tokenized_text , t[1]) for t in ans['explanations']['+']] + ['' for i in range(5-len(ans['explanations']['+']))]
	            exps_oppose = [positions2text(tokenized_text , t[1]) for t in ans['explanations']['-']] + ['' for i in range(5-len(ans['explanations']['-']))]
	            writer.writerow([example_id, method, json.dumps(exps_support), json.dumps(exps_oppose)])
	            utils.__log__(method)
	        print("-----------------------------------")