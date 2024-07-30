import gensim
import gensim.downloader
from gensim.models import KeyedVectors
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import paired_cosine_distances
import os

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")

word_vectors_names = ["glove-wiki-gigaword-50", "glove-wiki-gigaword-100", 
                    "glove-wiki-gigaword-200", "glove-wiki-gigaword-300", 
                    "word2vec-google-news-300"]

data_folder = "./dataset/vectors/"
create_folder_if_not_exists(data_folder)
for vector_name in word_vectors_names:
    word_vectors = gensim.downloader.load(vector_name)
    word_vectors.save(data_folder + vector_name)
