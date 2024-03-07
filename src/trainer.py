import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from logger import Logger
from tqdm import tqdm
from time import time

from model import Encoder
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from torch.optim.lr_scheduler import MultiplicativeLR


class BinaryEmbeddings(nn.Module):
    def __init__(self, opt):
        super(BinaryEmbeddings, self).__init__()
        self.num_bits = opt["num_bits"]
        self.lr = opt["lr"]
        self.epochs = opt["epochs"]
        self.device = opt["device"]
        self.embedding_size = opt["embedding_size"]
        self.lambd_weight_regularisation = opt["lambd_weight_regularisation"]
        self.lambd_cosine_regularisation = opt["lambd_cosine_regularisation"]
        self.lambd_john_regularisation = opt["lambd_john_regularisation"]
        self.lambd_sourav_regularisation = opt["lambd_sourav_regularisation"]

        self.encoder = Encoder(opt)
        self.loss = nn.MSELoss(reduction="mean")
        self.optimizer = torch.optim.SGD( filter(lambda p: p.requires_grad, self.encoder.parameters()), lr=self.lr, momentum=0.95)
        lmbda = lambda epoch: 0.1
        self.scheduler = MultiplicativeLR(self.optimizer, lr_lambda=lmbda)

    def process_batch(self, batch, train=True):
        
        word_embeddings_left = batch[0]
        word_embeddings_right = batch[1]
        encoder_input_left = word_embeddings_left.to(self.device)
        encoder_input_right = word_embeddings_right.to(self.device)
        binary_output_left, decoder_output_left, binary_output_right, decoder_output_right, \
            binary_cosine, binary_cosine_mean, true_cosine, true_cosine_blocks = self.encoder(encoder_input_left, encoder_input_right)

        autoencoder_loss_left = self.loss(encoder_input_left, decoder_output_left)
        autoencoder_loss_right = self.loss(encoder_input_right, decoder_output_right)
        binary_loss_john = self.loss(torch.exp(binary_cosine_mean), torch.exp(true_cosine))
        binary_loss_sourav = self.loss(torch.exp(binary_cosine), torch.exp(true_cosine_blocks))
        regularization_weight_encoder = torch.matmul(self.encoder.W_encoder, torch.t(self.encoder.W_encoder))
        regularization_weight_decoder = torch.matmul(self.encoder.W_decoder, torch.t(self.encoder.W_decoder))
        regularization_loss = 0.5 * torch.norm(regularization_weight_encoder - self.encoder.I) + 0.5 * torch.norm(regularization_weight_decoder - self.encoder.I)
        total_loss = 0.5*(autoencoder_loss_left + autoencoder_loss_right)+ self.lambd_weight_regularisation * regularization_loss \
                        + self.lambd_cosine_regularisation * (self.lambd_john_regularisation*binary_loss_john + self.lambd_sourav_regularisation*binary_loss_sourav)
        if train == True:
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        return binary_output_left, decoder_output_left, binary_output_right, decoder_output_right, total_loss.item()

    def train_model(self, trainDataLoader):
        self.train()
        for epoch in range(self.epochs):
            train_loss, cnt = 0, 0
            for idx, batch in tqdm(enumerate(trainDataLoader)):
                _, _, _, _, batch_loss= self.process_batch(batch, train=True)
                train_loss += batch_loss
                cnt+=1
                
            train_loss = train_loss/cnt
            # self.scheduler.step()
            print("Epoch = %d, Train Loss = %f"%(epoch+1, train_loss))
        self.scheduler = None
    
    def evaluate_model(self, testDataLoader):
        self.eval()
        test_loss, cnt = 0, 0
        cosine_preds, decoder_outputs, binary_outputs = [], [], []
        with torch.no_grad():
            for idx, batch in (enumerate(testDataLoader)):
                binary_output_left, decoder_output_left, binary_output_right, decoder_output_right, batch_loss = self.process_batch(batch, train=False)
                binary_output_left = binary_output_left.view(1, -1)
                binary_output_right = binary_output_right.view(1, -1)
                decoder_outputs.append(decoder_output_left.detach().to("cpu"))
                binary_outputs.append(binary_output_left.detach().to("cpu"))
                test_loss += batch_loss
                cnt+=1
            test_loss = test_loss/cnt
        print("Test Loss = %f"%(test_loss))
        decoder_outputs = torch.cat(decoder_outputs, dim=0)
        binary_outputs = torch.cat(binary_outputs, dim=0)
        return decoder_outputs, binary_outputs
