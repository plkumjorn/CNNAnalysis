#!/bin/bash

# Setup virtual environment and download required packages via pip
virtualenv -p /usr/bin/python3.6 venv
source venv/bin/activate
pip install -r requirements.txt

# Download spacy dependency
python -m spacy download en

# Download and install the newest version of keras-vis
git clone https://github.com/raghakot/keras-vis.git
cd keras-vis
python setup.py install
cd ..

# Create results and data directory 
mkdir -p results
mkdir -p data

# Download embeddings
cd data
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
rm glove.6B.zip

# Return to the main folder
cd ..