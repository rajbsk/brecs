from gensim.models import KeyedVectors
import numpy as np
import torch
import os

def load_embedding(embedding_name):
    word_vectors = KeyedVectors.load('dataset/vectors/' + embedding_name)
    return word_vectors

def load_dataset(word_vectors, dataset_name="MEN"):
    words_left, words_right, scores = [], [], []
    for line in open("dataset/"+dataset_name+".txt"):
        data = line[:-1].split("	")
        if data[0] not in word_vectors or data[1] not in word_vectors:
            continue
        words_left.append(data[0])
        words_right.append(data[1])
        scores.append(float(data[2]))

    scores = np.array(scores)
    if dataset_name == "MEN":
        scores = (scores - np.mean(scores, axis=-1))/(np.std(scores))

    return words_left, words_right, scores

def get_df_column_names(dataset_names):
    column_names = []
    for name in dataset_names:
        column_names.append(name+"_Decoder")
        column_names.append(name+"_Binary")
    return column_names

def cosine_scores(left_embeddings, right_embeddings):
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    output = cos(left_embeddings, right_embeddings)
    return output

def binary_scores(binary_left, binary_right):
    dim = binary_left.size()[-1]
    xnor = 1 - torch.abs(binary_left - binary_right)
    output = torch.sum(xnor, dim=-1)/dim
    return output

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")
