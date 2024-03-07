from xmlrpc.client import Boolean
import torch

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr, spearmanr
import os
from trainer import *
from utils import *
from dataset import *
import argparse
import pickle
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    embedding_name = args.embedding_name
    num_bits = args.num_bits
    train = args.is_train
    num_blocks = args.num_blocks
    lambd_weight_regularisation = args.lambd_weight_regularisation
    lambd_cosine_regularisation = args.lambd_cosine_regularisation
    lambd_john_regularisation = args.lambd_john_regularisation
    lambd_sourav_regularisation = args.lambd_sourav_regularisation
    
    
    model_directory ="models/"
    create_folder_if_not_exists(model_directory)
    model_name = "models/brecs_"+str(embedding_name)+"_"+str(num_bits)+"_l1"+str(lambd_weight_regularisation)+"_l2"+str(lambd_cosine_regularisation)+\
                    "_nb"+str(num_blocks)+"_jr"+str(lambd_john_regularisation)+"_sr"+str(lambd_sourav_regularisation)+".bin"
    word_vectors = load_embedding(embedding_name)
    print(model_name)
    opt = {"lr":0.001, "num_bits": num_bits, "batch_size": 256, "word_vectors": word_vectors, "epochs": 5, "device": device, 
            "num_blocks": num_blocks, "lambd_weight_regularisation": lambd_weight_regularisation, "lambd_cosine_regularisation": lambd_cosine_regularisation,
            "lambd_john_regularisation": lambd_john_regularisation, "lambd_sourav_regularisation":lambd_sourav_regularisation,
            "embedding_size": word_vectors.vector_size}

    if train == True:
        words = word_vectors.index_to_key
        dataset = [[random.choice(words), random.choice(words)] for _ in range(1000000)]
        dataset_train = EmbDataset(opt=opt, transform=transforms.Compose([ToTensor(opt)]), dataset=dataset)
        DatasetLoaderTrain = DataLoader(dataset_train, batch_size=opt["batch_size"], shuffle=True, num_workers=0, collate_fn=collate_fn)
        embedding_model = BinaryEmbeddings(opt)
        embedding_model = embedding_model.to(device)
        embedding_model.train_model(DatasetLoaderTrain)
        torch.save(embedding_model, model_name)
    else:
        embedding_model = torch.load(model_name)
        dataset_names = ["MEN", "RW", "SimLex", "WS353"]
        all_scores = []
        for dataset_name in dataset_names:
            words_left, words_right, scores = load_dataset(word_vectors, dataset_name=dataset_name)
            dataset_left = [[word, word] for word in words_left]
            dataset_right = [[word, word] for word in words_right]

            dataset_test_left = EmbDataset(opt=opt, transform=transforms.Compose([ToTensor(opt)]), dataset=dataset_left)
            DatasetLoaderTestLeft = DataLoader(dataset_test_left, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
            dataset_test_right = EmbDataset(opt=opt, transform=transforms.Compose([ToTensor(opt)]), dataset=dataset_right)
            DatasetLoaderRight = DataLoader(dataset_test_right, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
            decoder_outputs_left, binary_outputs_left = embedding_model.evaluate_model(DatasetLoaderTestLeft)
            decoder_outputs_right, binary_outputs_right = embedding_model.evaluate_model(DatasetLoaderRight)
            decoder_cosine_scores = cosine_scores(decoder_outputs_left, decoder_outputs_right)
            decoder_binary_scores = binary_scores(binary_outputs_left, binary_outputs_right)
            eval_spearman_cosine, _ = spearmanr(scores, decoder_cosine_scores.to("cpu"))
            eval_spearman_binary, _ = spearmanr(scores, decoder_binary_scores.to("cpu"))
            all_scores.append(eval_spearman_cosine)
            all_scores.append(eval_spearman_binary)
        print(all_scores)
        column_names = get_df_column_names(dataset_names)
        print(column_names)
        results = pd.DataFrame([all_scores], columns = column_names)
        results.to_csv("results/"+model_name.strip("models/")+".tsv", sep="\t", index=False)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_name', 
                        type=str, 
                        required=False,
                        default="glove-wiki-gigaword-300",
                        help='Name of the embedding model to be used.')
    parser.add_argument('--num_bits', 
                        type=int, 
                        required=True,
                        default=640,
                        help='Dataset Directory')
    parser.add_argument('--is_train', 
                        type=int,
                        required=True,
                        default=1,
                        help='Dataset Directory')
    parser.add_argument('--num_blocks', 
                        type=int,
                        required=True,
                        default=4,
                        help='Dataset Directory')
    parser.add_argument('--lambd_weight_regularisation', 
                        type=float,
                        required=True,
                        default=0.5,
                        help='Dataset Directory')
    parser.add_argument('--lambd_cosine_regularisation', 
                        type=float,
                        required=True,
                        default=0.5,
                        help='Dataset Directory')
    parser.add_argument('--lambd_john_regularisation', 
                        type=float,
                        required=True,
                        default=0.5,
                        help='Dataset Directory')
    parser.add_argument('--lambd_sourav_regularisation', 
                        type=float,
                        required=True,
                        default=0.5,
                        help='Dataset Directory')

    args = parser.parse_args()
    main(args)
