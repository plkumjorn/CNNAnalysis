#!/bin/bash

# This file should be called from the main folder using bash emnlp_experiments/setup.sh
# Create results and data directory 
mkdir -p results
mkdir -p data

# Download data for emnlp experiments
# Inside the data folder
cd data

# Amazon dataset
wget https://download1505.mediafire.com/t11wk5ogl8rg/4yspinakn96r7rz/amazonreviews.zip
unzip amazonreviews.zip
rm amazonreviews.zip

# ArXiv dataset
wget http://download2267.mediafire.com/dxmoins3m3tg/fv856z8uu7y03ug/arxiv.zip
unzip arxiv.zip
rm arxiv.zip

# Return to the main folder
cd ..