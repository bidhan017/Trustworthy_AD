d_min = 2
P = [1,2,3,4,4,5,6,6,7,8]
k = len(P)
for i in range(0, k):
    for j in range(i+1, k):
        reg = max(0, d_min - (P[i] - P[j])**2)**2
        print(reg)

        def divloss(self, prototype)
            k_list = []
            for i in range(prototype.shape[1]):
                for j in range(prototype.shape[0]):
                    l2 = max(0, d_min - np.power((prototype[batch, k] - prototype[batch, k]), 2)) ** 2
#                    l = nn.MSELoss(size_average=False, reduction='sum')(prototype[batch,k],enc_hidden[batch])
            return torch.sum(k_list)


        l2 = max(0, d_min - np.power((prototype[batch,k] - prototype[batch,k]), 2))**2



    ''''''''
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import multivariate_normal
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import trange
import math

from .algorithm_utils import Algorithm, PyTorchUtils

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, loss):
        if self.best_loss == None:
            self.best_loss = loss
        elif self.best_loss - loss > self.min_delta:
            self.best_loss = loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


class LSTMED(Algorithm, PyTorchUtils):
    def __init__(self, name: str = 'LSTM-ED', num_epochs: int = 10, batch_size: int = 64, lr: float = 0.5e-3,
                 hidden_size: int = 10, sequence_length: int = 1000, window: int =100, lamda: float = 1e-1 , train_gaussian_percentage: float = 0.30,
                 n_layers: tuple = (1, 1), use_bias: tuple = (True, True), dropout: tuple = (0, 0),
                 seed: int = None, gpu: int = None, details=True):
        Algorithm.__init__(self, __name__, name, seed, details=details)
        PyTorchUtils.__init__(self, seed, gpu)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr

        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.window = window
        self.train_gaussian_percentage = train_gaussian_percentage

        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout

        self.lamda = lamda

        self.lstmed = None
        self.mean, self.cov = None, None

    def fit(self, X: pd.DataFrame):
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        sequences = [data[i:i + self.sequence_length] for i in range(0, data.shape[0] - self.sequence_length + 1, self.window)]
        indices = np.random.permutation(len(sequences))
        split_point = int(self.train_gaussian_percentage * len(sequences))

        # train_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=True,
        #                           sampler=SubsetRandomSampler(indices), pin_memory=True)
        train_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=True,
                                  sampler=SubsetRandomSampler(indices[:-split_point]), pin_memory=True)
        train_gaussian_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=True,
                                            sampler=SubsetRandomSampler(indices[-split_point:]), pin_memory=True)

        self.lstmed = LSTMEDModule(X.shape[1], self.hidden_size,
                                   self.n_layers, self.use_bias, self.dropout,
                                   seed=self.seed, gpu=self.gpu)
        self.to_device(self.lstmed)
        optimizer = torch.optim.Adam(self.lstmed.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1, verbose =False)

        self.lstmed.train()

        epochloss=[]


        es = EarlyStopping(patience=5)

        for epoch in trange(self.num_epochs):
            logging.debug(f'Epoch {epoch+1}/{self.num_epochs}.')
            l=0.0
            for ts_batch in train_loader:
                output = self.lstmed(self.to_var(ts_batch))
                # optimizer.zero_grad()
                # loss = nn.MSELoss(size_average=False)(output, self.to_var(ts_batch.float()))
                loss = nn.L1Loss(reduce=True)(output, self.to_var(ts_batch.float()))

                L1_reg = torch.tensor(0., requires_grad=True)
                for name, param in self.lstmed.named_parameters():
                     if 'weight' in name:
                         L1_reg = L1_reg + torch.norm(param, 1)

               # L1_reg = torch.norm(self.lstmed.hidden2output.weight, 1)

                loss = loss + self.lamda * L1_reg
                l += loss.item()
                self.lstmed.zero_grad()
                loss.backward()
                optimizer.step()


            self.lstmed.eval()
            vl=0
            for ts_batch in train_gaussian_loader:
                output = self.lstmed(self.to_var(ts_batch))
                v_loss = nn.L1Loss(reduce=True)(output, self.to_var(ts_batch.float()))
                vl += v_loss.item()


            scheduler.step()
            epochloss.append(l/len(train_loader))
            print(f'Epoch {epoch+1} \t\t Training Loss: {l / len(train_loader)} \t\t Validation Loss: {vl / len(train_gaussian_loader)}')

            #Early stop
            es(vl / len(train_gaussian_loader))
            if es.early_stop:
                break

        self.lstmed.eval()
        error_vectors = []
        for ts_batch in train_gaussian_loader:
        # for ts_batch in train_loader:
            output = self.lstmed(self.to_var(ts_batch))
            error = nn.L1Loss(reduce=False)(output, self.to_var(ts_batch.float()))
            error_vectors += list(error.view(-1, X.shape[1]).data.cpu().numpy())

        self.mean = np.mean(error_vectors, axis=0)
        self.cov = np.cov(error_vectors, rowvar=False)

        return epochloss

    def predict(self, X: pd.DataFrame):
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        sequences = [data[i:i + self.sequence_length] for i in range(0, data.shape[0] - self.sequence_length + 1, self.window)]
        data_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, shuffle=False, drop_last=False)

        self.lstmed.eval()
        mvnormal = multivariate_normal(self.mean, self.cov, allow_singular=True)
        scores = []
        outputs = []
        errors = []
        for idx, ts in enumerate(data_loader):
            output = self.lstmed(self.to_var(ts))
            error = nn.L1Loss(reduce=False)(output, self.to_var(ts.float()))
            score = -mvnormal.logpdf(error.view(-1, X.shape[1]).data.cpu().numpy())
            scores.append(score.reshape(ts.size(0), self.sequence_length))
            if self.details:
                outputs.append(output.data.numpy())
                errors.append(error.data.numpy())

        # stores seq_len-many scores per timestamp and averages them
        scores = np.concatenate(scores)
        lattice = np.full((self.sequence_length, data.shape[0]), np.nan)
        for i, score in enumerate(scores):
            lattice[i % self.sequence_length, (i*self.window): (i*self.window) + self.sequence_length] = score
        scores = np.nanmean(lattice, axis=0)
        # scores = np.nan_to_num(scores) #added

        if self.details:
            outputs = np.concatenate(outputs)
            lattice = np.full((self.sequence_length, X.shape[0], X.shape[1]), np.nan)
            for i, output in enumerate(outputs):
                lattice[i % self.sequence_length, (i*self.window): (i*self.window) + self.sequence_length, :] = output
            # self.prediction_details.update({'reconstructions_mean': np.nan_to_num(np.nanmean(lattice, axis=0).T)}) #added
            self.prediction_details.update({'reconstructions_mean': (np.nanmean(lattice, axis=0).T)})

            errors = np.concatenate(errors)
            lattice = np.full((self.sequence_length, X.shape[0], X.shape[1]), np.nan)
            for i, error in enumerate(errors):
                lattice[i % self.sequence_length, (i*self.window): (i*self.window) + self.sequence_length, :] = error
            # self.prediction_details.update({'errors_mean': np.nan_to_num(np.nanmean(lattice, axis=0).T)}) #added
            self.prediction_details.update({'errors_mean': (np.nanmean(lattice, axis=0).T)})

        return scores


class Prototype(nn.Module):

    def __init__(self, k, hidden_size):
        super().__init__()
        self.k = k
        self.hidden_size = hidden_size
        weights = torch.Tensor(k, hidden_size)
        self.weights = nn.Parameter(weights, requires_grad = True)  # nn.Parameter is a Tensor that's a module parameter.


        # initialize weights and biases
        nn.init.normal_(self.weights, mean=0, std=1.0) # weight init
        # fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        # bound = 1 / math.sqrt(fan_in)

    def forward(self, x):

        d = torch.norm( x-self.weights, dim=0)
        return torch.exp(-d)

# class Prototype(nn.Module):
#     """ Custom Linear layer but mimics a standard linear layer """
#     def __init__(self, size_in, size_out):
#         super().__init__()
#         self.size_in, self.size_out = size_in, size_out
#         weights = torch.Tensor(size_out, size_in)
#         self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
#         bias = torch.Tensor(size_out)
#         self.bias = nn.Parameter(bias)

#         # initialize weights and biases
#         nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init
#         fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
#         bound = 1 / math.sqrt(fan_in)
#         nn.init.uniform_(self.bias, -bound, bound)  # bias init

#     def forward(self, x):
#         w_times_x= torch.mm(x, self.weights.t())
#         return torch.add(w_times_x, self.bias)  # w times x + b

class LSTMEDModule(nn.Module, PyTorchUtils):
    def __init__(self, n_features: int, hidden_size: int,
                 n_layers: tuple, use_bias: tuple, dropout: tuple,
                 seed: int, gpu: int):
        super().__init__()
        PyTorchUtils.__init__(self, seed, gpu)
        self.n_features = n_features
        self.hidden_size = hidden_size

        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout

        self.encoder = nn.GRU(self.n_features, self.hidden_size, batch_first=True,
                               num_layers=self.n_layers[0], bias=self.use_bias[0], dropout=self.dropout[0])
        self.to_device(self.encoder)

        self.decoder = nn.GRU(self.n_features, self.hidden_size, batch_first=True,
                               num_layers=self.n_layers[1], bias=self.use_bias[1], dropout=self.dropout[1])
        self.to_device(self.decoder)

        self.prototype = Prototype(10, self.hidden_size)
        self.to_device(self.prototype)

        self.hidden2output = nn.Linear(self.hidden_size, self.n_features)
        self.to_device(self.hidden2output)

    def _init_hidden(self, batch_size):
        return self.to_var(torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_())
        # return (self.to_var(torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_()),
        #         self.to_var(torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_()))

    def forward(self, ts_batch, return_latent: bool = False):
        batch_size = ts_batch.shape[0]

        # 1. Encode the timeseries to make use of the last hidden state.
        enc_hidden = self._init_hidden(batch_size)  # initialization with zero
        _, enc_hidden = self.encoder(ts_batch.float(), enc_hidden)  # .float() here or .double() for the model

        # 2. Use hidden state as initialization for our Decoder-LSTM
        dec_hidden = enc_hidden

        # 3. Also, use this hidden state to get the first output aka the last point of the reconstructed timeseries
        # 4. Reconstruct timeseries backwards
        #    * Use true data for training decoder
        #    * Use hidden2output for prediction
        output = self.to_var(torch.Tensor(ts_batch.size()).zero_())
        for i in reversed(range(ts_batch.shape[1])):
            interout = self.prototype(dec_hidden[0][0, :])
            # output[:, i, :] = self.Prototype(dec_hidden[0][0, :].expand(1,dec_hidden[0][0, :].shape[0]))
            output[:, i, :] = self.hidden2output(interout)
#
            if self.training:
                _, dec_hidden = self.decoder(ts_batch[:, i].unsqueeze(1).float(), dec_hidden)
            else:
                _, dec_hidden = self.decoder(output[:, i].unsqueeze(1), dec_hidden)

        return (output, enc_hidden[1][-1]) if return_latent else output
    
    '''