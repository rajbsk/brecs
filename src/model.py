import torch
import torch.nn.functional as F
from torch import nn
import math

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
    
class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x    

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.ste = StraightThroughEstimator()
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        self.num_bits = opt["num_bits"]
        self.device = opt["device"]
        self.embedding_size = opt["embedding_size"]

        # Identity Matrix
        self.I = (torch.eye(self.embedding_size)).to(self.device)
        self.heav_val = (torch.tensor(0.0)).to(self.device)
        self.ste = StraightThroughEstimator()

        # cosine vector
        self.cosine_vector = torch.tensor([math.pow(2, -i) for i in range(self.num_bits)])
        self.cosine_vector = ((torch.tensor(self.cosine_vector)).unsqueeze(0)).to(self.device)
        
        # Network Layers:
        self.W_encoder = torch.nn.Parameter(torch.randn(self.embedding_size, self.num_bits))
        self.W_encoder.requires_grad = True
        self.W_decoder = torch.nn.Parameter(torch.randn(self.embedding_size, self.num_bits))
        self.W_decoder.requires_grad = True

    
    def forward(self, word_embeddings_left, word_embeddings_right):
        encoder_output_left = torch.matmul(word_embeddings_left, self.W_encoder)
        binary_output_left = self.ste(encoder_output_left)
        decoder_output_left = torch.matmul(binary_output_left, torch.t(self.W_decoder))
        decoder_output_left = F.tanh(decoder_output_left)

        encoder_output_right = torch.matmul(word_embeddings_right, self.W_encoder)
        binary_output_right = self.ste(encoder_output_right)
        decoder_output_right = torch.matmul(binary_output_right, torch.t(self.W_decoder))
        decoder_output_right = F.tanh(decoder_output_right)
    
        xnor_output = 1 - torch.abs(binary_output_left - binary_output_right)
        binary_cosine = torch.sum((xnor_output*self.cosine_vector), dim=-1)

        true_cosine = self.cosine_similarity(word_embeddings_left, word_embeddings_right) + 1
        word_embeddings_left = F.normalize(word_embeddings_left, dim=-1)
        word_embeddings_right = F.normalize(word_embeddings_right, dim=-1)

        return binary_output_left, decoder_output_left, binary_output_right, decoder_output_right, binary_cosine, true_cosine


