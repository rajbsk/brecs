from xmlrpc.client import Boolean
import torch
torch.manual_seed(42)
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr, spearmanr


from trainer import *
from utils import *
from dataset import *
import argparse
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def main(args):
    dataset_name = args.dataset_name
    num_bits = args.num_bits
    train = args.is_train
    num_blocks = args.num_blocks
    lambd_weight_regularisation = args.lambd_weight_regularisation
    lambd_cosine_regularisation = args.lambd_cosine_regularisation
    lambd_john_regularisation = args.lambd_john_regularisation
    lambd_sourav_regularisation = args.lambd_sourav_regularisation
    # dataset_name = "MEN"
    # num_bits = 640
    # train = 1
    # num_blocks = 40
    model_name = "models/w2v_aaai_ste_now_ep5_john_sourav_"+str(num_bits)+"_l1"+str(lambd_weight_regularisation)+"_l2"+str(lambd_cosine_regularisation)+\
                    "_nb"+str(num_blocks)+"_jr"+str(lambd_john_regularisation)+"_sr"+str(lambd_sourav_regularisation)+".bin"
    word_vectors = load_w2v()

    opt = {"lr":0.001, "num_bits": num_bits, "batch_size": 256, "word_vectors": word_vectors, "epochs": 5, "device": device, 
            "num_blocks": num_blocks, "lambd_weight_regularisation": lambd_weight_regularisation, "lambd_cosine_regularisation": lambd_cosine_regularisation,
            "lambd_john_regularisation": lambd_john_regularisation, "lambd_sourav_regularisation":lambd_sourav_regularisation}
    dataset = []

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

        # Fetch binary word embeddings from the glove vectors.
        # glove_words = word_vectors.index_to_key
        # dataset_glove = [[word, word] for word in glove_words]
        # glove_words = dataset_glove
        # dataset_glove_words = EmbDataset(opt=opt, transform=transforms.Compose([ToTensor(opt)]), dataset=glove_words)
        # DatasetLoaderGlove = DataLoader(dataset_glove_words, batch_size=1000, shuffle=False, num_workers=0, collate_fn=collate_fn)
        # _, binary_outputs_glove = embedding_model.evaluate_model(DatasetLoaderGlove)
        # siz = binary_outputs_glove.size()[0]
        # binary_outputs_glove = binary_outputs_glove.view(siz, -1)
        # binary_outputs_glove = (binary_outputs_glove.to("cpu")).to(torch.int8)
        # print(binary_outputs_glove.size())
        # embedding_dict = {}
        # for i in range(len(glove_words)):
        #     word = glove_words[i][1]
        #     embedding_dict[word] = binary_outputs_glove[i].tolist()
        # with open("embeddings/aaai_binary_embedding_siamese_decoupled_w2v_lw0.3_lcs_0.7_lj0.2_ls0.8.pkl", "wb") as f:
        #     pickle.dump(embedding_dict, f)
        # exit()

        print(dataset_name)
        words_left, words_right, scores = load_dataset(word_vectors, dataset_name=dataset_name)
        dataset_left = [[word, word] for word in words_left]
        dataset_right = [[word, word] for word in words_right]

        dataset_test_left = EmbDataset(opt=opt, transform=transforms.Compose([ToTensor(opt)]), dataset=dataset_left)
        DatasetLoaderTestLeft = DataLoader(dataset_test_left, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
        dataset_test_right = EmbDataset(opt=opt, transform=transforms.Compose([ToTensor(opt)]), dataset=dataset_right)
        DatasetLoaderRight = DataLoader(dataset_test_right, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
        embedding_model = torch.load(model_name)
        decoder_outputs_left, binary_outputs_left = embedding_model.evaluate_model(DatasetLoaderTestLeft)
        decoder_outputs_right, binary_outputs_right = embedding_model.evaluate_model(DatasetLoaderRight)
        decoder_cosine_scores = cosine_scores(decoder_outputs_left, decoder_outputs_right)
        decoder_binary_scores = binary_scores(binary_outputs_left, binary_outputs_right)
        eval_spearman_cosine, _ = spearmanr(scores, decoder_cosine_scores.to("cpu"))
        eval_spearman_binary, _ = spearmanr(scores, decoder_binary_scores.to("cpu"))
        print("Spearman scores Decoder:")
        print(eval_spearman_cosine)
        print("Spearman scores Binary:")
        print(eval_spearman_binary)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', 
                        type=str, 
                        required=False,
                        default="MEN",
                        help='Path to the training dataset text file')
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
                        default=40,
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
    # args = None
    main(args)
