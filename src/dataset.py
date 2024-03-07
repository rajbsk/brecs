import torch
import random
from torch.utils.data import Dataset, DataLoader

def collate_fn(batch):
    word_embeddings_left = []
    word_embeddings_right = []
    for sample in batch:
        word_embeddings_left.append(sample[0])
        word_embeddings_right.append(sample[1])
    word_embeddings_left = torch.stack(word_embeddings_left)
    word_embeddings_right = torch.stack(word_embeddings_right)
    
    return word_embeddings_left, word_embeddings_right


class EmbDataset(Dataset):
    def __init__(self, opt, transform, dataset):
        self.transform = transform
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.dataset[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

class ToTensor(object):
    def __init__(self, opt):
        self.word_vectors = opt["word_vectors"]

    def __call__(self, sample):
        word_left = sample[0]
        word_right = sample[1]
        word_embedding_left = torch.tensor(self.word_vectors[word_left])
        word_embedding_right = torch.tensor(self.word_vectors[word_right])
        return word_embedding_left, word_embedding_right
