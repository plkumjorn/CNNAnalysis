import sys
sys.path.append("..")
from analysis import *
from pprint import pprint
import json, csv

# Project name
result_folder = "../results/"
project_name = "4Newsgroups"
model_name = 'model1'

# Download embedding 
emb_matrix, vocab_size, emb_dim, max_len, word_index, word2index = get_embedding_matrix(EMBEDDINGS_PATH['glove-300'], max_len = 150)

# Download 20Newsgroup data
target_names, text_train, y_train, text_validate, y_validate, text_test, y_test = load_20newsgroups(ratio = [0.6, 0.2, 0.2], remove=('headers', 'footers'), categories = ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'])

# Data preparation
utils.__log__("Start prepare training data")
X_train, y_train = get_data_matrix(text_train, word2index, max_len), y_train
utils.__log__("Start prepare validation data")
X_validate, y_validate = get_data_matrix(text_validate, word2index, max_len), y_validate
utils.__log__("Start prepare testing data")
X_test, y_test = get_data_matrix(text_test, word2index, max_len), y_test
utils.__log__("Finish")
y_train_onehot, y_validate_onehot, y_test_onehot = to_categorical(y_train), to_categorical(y_validate), to_categorical(y_test) 
understand_data(target_names, y_train, y_test, y_validate)

# Model creation and training
cnn_model = CNNModel(vocab_size, word_index, word2index, emb_dim, emb_matrix, False, max_len, target_names, \
                     filters = [(2, 50), (3, 50), (4, 50)], \
                     filter_activations = 'relu', \
                     dense = [150, len(target_names)], \
                     dense_activations = ['relu', 'softmax'])

project_path = result_folder + project_name
cnn_model.train(project_path, model_name, X_train, y_train_onehot, X_validate, y_validate_onehot)

# Test the model
prediction_test = cnn_model.predict(X_test, batch_size = 128)
print(classification_report(y_test, prediction_test, target_names=target_names))

# Explanation examples
# 1) Select an example
index = 32
input_text = text_test[index]
print(f"Input test: {input_text}")

# 2) Predict
fe_input = utils.get_data_matrix([input_text], word2index, cnn_model.max_len, use_tqdm = False)[0]
predicted_class = cnn_model.predict(np.array([fe_input]))
print(f"The predicted class is {target_names[predicted_class]} (class_id = {predicted_class})")
print(f"The actual class is {target_names[y_test[index]]} (class_id = {y_test[index]})")

# 3) Local explanations
k = 5 # Number of texts for an explanation
for method in ['lime', 'lrp_words', 'lrp_ngrams', 'decision_trees', 'grad_cam']:
    print('-----------------------------------------------------')
    ans = explain_example(cnn_model, input_text, method, count = k)
    print(method)
    print(f"Top {k} evidence texts: {len(ans['explanations']['+'])} texts returned")
    for tup in ans['explanations']['+']:
    	print(f"\t{tup[0]}")
    print(f"Top {k} counter-evidence texts: {len(ans['explanations']['-'])} texts returned")
    for tup in ans['explanations']['-']:
    	print(f"\t{tup[0]}")
    utils.__log__(method)
    