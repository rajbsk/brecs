import gensim
import gensim.downloader
from gensim.models import KeyedVectors
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import paired_cosine_distances

def load_dataset_file(dataset_name, word_vectors):
    words_left, words_right, scores = [], [], []
    for line in open("./dataset/"+dataset_name+".txt"):
        data = line[:-1].split("	")
        if data[0] not in word_vectors or data[1] not in word_vectors:
            continue
        words_left.append(data[0])
        words_right.append(data[1])
        scores.append(float(data[2]))
    return words_left, words_right, scores

def main():
    word_vectors_names = ["glove-wiki-gigaword-50", "glove-wiki-gigaword-100", 
                        "glove-wiki-gigaword-200", "glove-wiki-gigaword-300", 
                        "word2vec-google-news-300"]
    dataset_names = ["MEN", "RW", "SimLex", "WS353"]
    all_scores = []
    for word_vectors_name in word_vectors_names:
        word_vectors = KeyedVectors.load('./dataset/vectors/'+word_vectors_name)
        instance_scores = []
        for dataset_name in dataset_names:
            words_left, words_right, scores = load_dataset_file(dataset_name, word_vectors)

            scores = np.array(scores)
            if dataset_name == "MEN":
                scores = (scores - np.mean(scores, axis=-1))/(np.std(scores))

            word_vectors_left = torch.stack([torch.tensor(word_vectors[word]) for word in words_left])
            word_vectors_right = torch.stack([torch.tensor(word_vectors[word]) for word in words_right])

            cosine_scores = 1 - (paired_cosine_distances(word_vectors_left, word_vectors_right))
            eval_spearman_cosine, _ = spearmanr(scores, cosine_scores)
            instance_scores.append(eval_spearman_cosine*100)
        all_scores.append(instance_scores)
    results = pd.DataFrame(all_scores, index=word_vectors_names, columns = dataset_names)
    results.to_csv("results/vanilla_embeddings.tsv", sep="\t")
            
if __name__=="__main__":
    main()
