# EMBEDDINGS_PATH is a dictionary of embedding names and the corresponding paths in your local computer
# Only two formats are accepted at present -- .txt (such as glove) and .pickle (containing a dictionary of words and their vectors)
EMBEDDINGS_PATH = { 'glove-200':'../data/glove.6B.200d.txt',
					'glove-300':'../data/glove.6B.300d.txt'}

# GRAPHVIZ_PATH is a path of the folder of installed graphviz
GRAPHVIZ_PATH = ''
# GRAPHVIZ_PATH = 'C:\\Program Files (x86)\\Graphviz2.38\\bin'

# A parameter used in the cnn_fa method
PURITY_CNNFA = 0.4