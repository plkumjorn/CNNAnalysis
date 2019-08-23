
# coding: utf-8

# # ArXiv Datasets

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
project_name = 'ArXiv'
project_path = result_folder + project_name
model_name_general = 'ArXiv_general'
model_name_specific = 'ArXiv_specific'

# ## Download the datasets
# - ArXiv datasets
target_names = ['cs', 'math', 'phys']
METHODS = ['random_words', 'random_ngrams', 'lime', 'lrp_words', 'lrp_ngrams', 'deeplift_words', 'deeplift_ngrams','decision_trees', 'grad_cam']
# METHODS = ['random_words', 'random_ngrams', 'lime', 'decision_trees', 'grad_cam']

folder_path = "../data/arxiv/"

text_train_general = pickle.load(open(folder_path + "text_train_general.pickle", "rb"))
text_train_specific = pickle.load(open(folder_path + "text_train_specific.pickle", "rb"))
text_validate_general = pickle.load(open(folder_path + "text_validate_general.pickle", "rb"))
text_validate_specific = pickle.load(open(folder_path + "text_validate_specific.pickle", "rb"))
text_test_general = pickle.load(open(folder_path + "text_test_general.pickle", "rb"))

y_train_general = pickle.load(open(folder_path + "y_train_general.pickle", "rb"))
y_train_specific = pickle.load(open(folder_path + "y_train_specific.pickle", "rb"))
y_validate_general = pickle.load(open(folder_path + "y_validate_general.pickle", "rb"))
y_validate_specific = pickle.load(open(folder_path + "y_validate_specific.pickle", "rb"))
y_test_general = pickle.load(open(folder_path + "y_test_general.pickle", "rb"))

# Preprocessing

X_train_general, X_validate_general = get_data_matrix(text_train_general, word2index, max_len), get_data_matrix(text_validate_general, word2index, max_len)
X_train_specific, X_validate_specific = get_data_matrix(text_train_specific, word2index, max_len), get_data_matrix(text_validate_specific, word2index, max_len)
X_test_general = get_data_matrix(text_test_general, word2index, max_len)

y_train_general_onehot, y_validate_general_onehot, y_test_general_onehot = to_categorical(y_train_general), to_categorical(y_validate_general), to_categorical(y_test_general) 
y_train_specific_onehot, y_validate_specific_onehot = to_categorical(y_train_specific), to_categorical(y_validate_specific)
understand_data(target_names, y_train_general, y_test_general, y_validate_general)

# ## Model creation and training

# ### Model 1: Train until the validation loss does not improve for three epochs
cnn_model = CNNModel(vocab_size, word_index, word2index, emb_dim, emb_matrix, False, max_len, target_names,                      filters = [(2, 50), (3, 50), (4, 50)],                      filter_activations = 'relu',                      dense = [150, len(target_names)],                      dense_activations = ['relu', 'softmax'])

cnn_model.train(project_path, model_name_general, X_train_general, y_train_general_onehot, X_validate_general, y_validate_general_onehot, batch_size = 2048)

# ### Model 2: Train for only one epoch

cnn_model_rush = CNNModel(vocab_size, word_index, word2index, emb_dim, emb_matrix, False, max_len, target_names,                      filters = [(2, 50), (3, 50), (4, 50)],                      filter_activations = 'relu',                      dense = [150, len(target_names)],                      dense_activations = ['relu', 'softmax'])

cnn_model_rush.train(project_path, model_name_specific, X_train_specific, y_train_specific_onehot, X_validate_specific, y_validate_specific_onehot, batch_size = 2048)


# ## Evaluating both models

# - With ArXiv general evaluation

prediction_test_onehot = cnn_model.predict(X_test_general, batch_size = 128, one_hot = True)
print(classification_report(y_test_general, prediction_test_onehot.argmax(axis=1).squeeze(), target_names=target_names))


prediction_test_onehot_rush = cnn_model_rush.predict(X_test_general, batch_size = 128, one_hot = True)
print(classification_report(y_test_general, prediction_test_onehot_rush.argmax(axis=1).squeeze(), target_names=target_names))

# Decision tree evaluations
# DTs for the well-trained CNN
prediction_test = prediction_test_onehot.argmax(axis=1).squeeze()
feature_test_matrix = cnn_model.feature_extraction_model.predict(X_test_general)
prediction_test_byprunedtreelist = predict_from_treelist(cnn_model.pruned_tree_list, feature_test_matrix)
print("Fidelity: ")
print(classification_report(prediction_test, prediction_test_byprunedtreelist, target_names=target_names))
print("Accuracy: ")
print(classification_report(y_test_general, prediction_test_byprunedtreelist, target_names=target_names))
print("Tree stats:")
for idx, t in enumerate(tqdm(cnn_model.pruned_tree_list)):
    n, d, l = tree_stats(t.tree_, 0)
    print("{}: Node {}, Depth {}, Leaves {}".format(target_names[idx], n, d, l))
pickle.dump(cnn_model.pruned_tree_list, open(project_path + '/' + model_name_general + '_pruned_tree_list.pickle', 'wb'))
draw_tree_list(cnn_model.pruned_tree_list, cnn_model, folder = project_path)
print("-----------------------------------------------------------------")

# DTs for the worse CNN
prediction_test_rush = prediction_test_onehot_rush.argmax(axis=1).squeeze()
print(classification_report(y_test_general, prediction_test_rush, target_names=target_names))
feature_test_matrix_rush = cnn_model_rush.feature_extraction_model.predict(X_test_general)
prediction_test_byprunedtreelist_rush = predict_from_treelist(cnn_model_rush.pruned_tree_list, feature_test_matrix_rush)
print("Fidelity: ")
print(classification_report(prediction_test_rush, prediction_test_byprunedtreelist_rush, target_names=target_names))
print("Accuracy: ")
print(classification_report(y_test_general, prediction_test_byprunedtreelist_rush, target_names=target_names))
print("Tree stats:")
for idx, t in enumerate(tqdm(cnn_model_rush.pruned_tree_list)):
    n, d, l = tree_stats(t.tree_, 0)
    print("{}: Node {}, Depth {}, Leaves {}".format(target_names[idx], n, d, l))
pickle.dump(cnn_model_rush.pruned_tree_list, open(project_path + '/' + model_name_specific + '_pruned_tree_list.pickle', 'wb'))
draw_tree_list(cnn_model_rush.pruned_tree_list, cnn_model_rush, folder = project_path)


# ## User Study A: Trustworthiness

# - Select 100 test texts for testing
#     - 50 texts : both CNNs correctly classify (mixed)
#     - 50 texts : both CNNs misclassify (mixed)

CORRECT, INCORRECT = set([]), set([])
for example_idx, y in enumerate(tqdm(y_test_general)):
    if prediction_test[example_idx] == prediction_test_rush[example_idx]:
        if y == prediction_test[example_idx]: # Correct classify
            CORRECT.add(example_idx)
        else: # Misclassify
            INCORRECT.add(example_idx)
print([len(aset) for aset in [CORRECT, INCORRECT]])
CORRECT, INCORRECT = list(CORRECT), list(INCORRECT)
random.shuffle(CORRECT); random.shuffle(INCORRECT);
CORRECT, INCORRECT = CORRECT[:50], INCORRECT[:50] 


# - Write tokenized texts into a file

with open(project_path + '/user_study_a_tokenizedtexts.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file, lineterminator="\n")
    writer.writerow(['example_id', 'text', 'tokenized_text', 'actual_class', 'predicted_class'])
    for alist in [CORRECT, INCORRECT]:
        for example_id in alist:
            tokenized_text = list(tokenizer(text_test_general[example_id]))
            tokenized_text = [str(w) for w in tokenized_text]
            writer.writerow([example_id, text_test_general[example_id], json.dumps(tokenized_text), y_test_general[example_id], prediction_test[example_id]])


# - Dump explanations to a file

# # CNN Model full time training
with open(project_path + '/user_study_a_explanations_specific.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file, lineterminator="\n")
    writer.writerow(['example_id', 'explain_method', 'ngrams'])
    for alist in [CORRECT, INCORRECT]:
        for idx, example_id in enumerate(tqdm(alist)):
            print(idx, '; Example_id', example_id)
            for method in METHODS:
                ans = explain_example(cnn_model, text_test_general[example_id], method, exp_type = ['+'])
                exps = [list(t[1]) for t in ans['explanations']['+']] + [list([]) for i in range(5-len(ans['explanations']['+']))]
                exps = [[int(idx) for idx in ng] for ng in exps]
                writer.writerow([example_id, method, json.dumps(exps)])
                utils.__log__(method)
            print("-----------------------------------")

# # CNN Model rush time training
with open(project_path + '/user_study_a_explanations_specific.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file, lineterminator="\n")
    writer.writerow(['example_id', 'explain_method', 'ngrams'])
    for alist in [CORRECT, INCORRECT]:
        for example_id in tqdm(alist):
            for method in METHODS:
                ans = explain_example(cnn_model_rush, text_test_general[example_id], method, exp_type = ['+'])
                exps = [list(t[1]) for t in ans['explanations']['+']] + [list([]) for i in range(5-len(ans['explanations']['+']))]
                exps = [[int(idx) for idx in ng] for ng in exps]
                writer.writerow([example_id, method, json.dumps(exps)])
                utils.__log__(method)
            print("-----------------------------------")

# ## User Study B: Class Discriminative

# - Using only the good CNN, select 100 test texts for testing
#     - 50 texts : The CNN correctly classifies with 90% confidence or more (mixed)
#     - 50 texts : The CNN misclassifies with 90% confidence or more (mixed)

CORRECT, INCORRECT = set([]), set([])
is_correct = prediction_test == y_test_general
max_confidence = np.max(prediction_test_onehot, axis = 1)
for example_idx, y in enumerate(tqdm(y_test_general)):
    if max_confidence[example_idx] > 0.9:
        if is_correct[example_idx]: # Correct classify
            CORRECT.add(example_idx)
        else: # Misclassify
            INCORRECT.add(example_idx)
print([len(aset) for aset in [CORRECT, INCORRECT]])
CORRECT, INCORRECT = list(CORRECT), list(INCORRECT)
random.shuffle(CORRECT); random.shuffle(INCORRECT);
CORRECT, INCORRECT = CORRECT[:50], INCORRECT[:50] 


# # - Write texts into a file

with open(project_path + '/user_study_b_texts.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file, lineterminator="\n")
    writer.writerow(['example_id', 'text', 'actual_class', 'predicted_class'])
    for alist in [CORRECT, INCORRECT]:
        for example_id in alist:
            writer.writerow([example_id, text_test_general[example_id], y_test_general[example_id], prediction_test[example_id]])


# - Dump explanations to a file

# # # CNN Model full time training
with open(project_path + '/user_study_b_explanations_full.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file, lineterminator="\n")
    writer.writerow(['example_id', 'explain_method', 'ngrams'])
    for alist in [CORRECT, INCORRECT]:
        for example_id in tqdm(alist):
            input_text = text_test_general[example_id]
            tokenized_text = [str(w) for w in list(utils.tokenizer(input_text)) if str(w) != '']
            for method in METHODS:
                ans = explain_example(cnn_model, text_test_general[example_id], method, exp_type = ['+'])
                exps = [positions2text(tokenized_text , t[1]) for t in ans['explanations']['+']] + ['' for i in range(5-len(ans['explanations']['+']))]
                writer.writerow([example_id, method, json.dumps(exps)])
                utils.__log__(method)
            print("-----------------------------------")

# ## User Study C: Providing Information for Decision Making

# - Using only the good CNN, select 100 test texts for testing
#     - 50 texts : The CNN correctly classifies with 70% confidence or less (mixed)
#     - 50 texts : The CNN misclassifies with 70% confidence or less (mixed)

CORRECT, INCORRECT = set([]), set([])
is_correct = prediction_test == y_test_general
max_confidence = np.max(prediction_test_onehot, axis = 1)
for example_idx, y in enumerate(tqdm(y_test_general)):
    if max_confidence[example_idx] < 0.7:
        if is_correct[example_idx]: # Correct classify
            CORRECT.add(example_idx)
        else: # Misclassify
            INCORRECT.add(example_idx)
print([len(aset) for aset in [CORRECT, INCORRECT]])
CORRECT, INCORRECT = list(CORRECT), list(INCORRECT)
random.shuffle(CORRECT); random.shuffle(INCORRECT);
CORRECT, INCORRECT = CORRECT[:50], INCORRECT[:50] 


# # - Write texts into a file

with open(project_path + '/user_study_c_texts_with_scores.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file, lineterminator="\n")
    writer.writerow(['example_id', 'text', 'actual_class', 'predicted_class', 'scores'])
    for alist in [CORRECT, INCORRECT]:
        for example_id in alist:
            scores = list(prediction_test_onehot[example_id])
            scores = [round(float(s), 3) for s in scores]
            writer.writerow([example_id, text_test_general[example_id], y_test_general[example_id], prediction_test[example_id], json.dumps(scores)])



# - Dump explanations to a file

# CNN Model full time training
with open(project_path + '/user_study_c_explanations_full.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file, lineterminator="\n")
    writer.writerow(['example_id', 'explain_method', 'ngrams_support', 'ngrams_oppose'])
    for alist in [CORRECT, INCORRECT]:
        for example_id in alist:
            input_text = text_test_general[example_id]
            tokenized_text = [str(w) for w in list(utils.tokenizer(input_text)) if str(w) != '']
            for method in METHODS:
                ans = explain_example(cnn_model, text_test_general[example_id], method, exp_type = ['+', '-'])
                exps_support = [positions2text(tokenized_text , t[1]) for t in ans['explanations']['+']] + ['' for i in range(5-len(ans['explanations']['+']))]
                exps_oppose = [positions2text(tokenized_text , t[1]) for t in ans['explanations']['-']] + ['' for i in range(5-len(ans['explanations']['-']))]
                writer.writerow([example_id, method, json.dumps(exps_support), json.dumps(exps_oppose)])
                utils.__log__(method)
            print("-----------------------------------")

