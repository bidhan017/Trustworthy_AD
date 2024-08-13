import glob
import os

import numpy as np
import pandas as pd

from src.algorithms import LSTMED
from src.datasets import RealPickledDataset
from src.evaluation import Evaluator

RUNS = 1


def main():
    evaluate_real_datasets()


def detectors(seed,step,sequence_length):
    standard_epochs = 40
    dets = [
            LSTMED(num_epochs=standard_epochs, seed=seed, step=step, sequence_length=sequence_length)
            ]

    return sorted(dets, key=lambda x: x.framework)


'''def run_experiments():
    # Set the seed manually for reproducibility.
    seeds = np.random.randint(np.iinfo(np.uint32).max, size=RUNS, dtype=np.uint32)
    output_dir = 'reports/experiments'
    evaluators = []
    outlier_height_steps = 10
    for outlier_type in ['extreme_1', 'shift_1', 'variance_1', 'trend_1']:
        announce_experiment('Outlier Height')
        ev_extr = run_extremes_experiment(
            detectors, seeds, RUNS, outlier_type, steps=outlier_height_steps,
            output_dir=os.path.join(output_dir, outlier_type, 'intensity'))
        evaluators.append(ev_extr)
    announce_experiment('Multivariate Datasets')
    ev_mv = run_multivariate_experiment(
        detectors, seeds, RUNS,
        output_dir=os.path.join(output_dir, 'multivariate'))
    evaluators.append(ev_mv)
    for mv_anomaly in ['doubled', 'inversed', 'shrinked', 'delayed', 'xor', 'delayed_missing']:
        announce_experiment(f'Multivariate Polluted {mv_anomaly} Datasets')
        ev_mv = run_multivariate_polluted_experiment(
            detectors, seeds, RUNS, mv_anomaly,
            output_dir=os.path.join(output_dir, 'mv_polluted'))
        evaluators.append(ev_mv)
        announce_experiment(f'High-dimensional multivariate {mv_anomaly} outliers')
        ev_mv_dim = run_multi_dim_multivariate_experiment(
            detectors, seeds, RUNS, mv_anomaly, steps=20,
            output_dir=os.path.join(output_dir, 'multi_dim_mv'))
        evaluators.append(ev_mv_dim)
    announce_experiment('Long-Term Experiments')
    ev_different_windows = run_different_window_sizes_evaluator(different_window_detectors, seeds, RUNS)
    evaluators.append(ev_different_windows)
    for ev in evaluators:
        ev.plot_single_heatmap()
'''

def evaluate_real_datasets():
    REAL_DATASET_GROUP_PATH = 'data/raw/'
    real_dataset_groups = glob.glob(REAL_DATASET_GROUP_PATH + '*')
    seeds = [3424441230]
    results = pd.DataFrame()
    datasets = []
    for real_dataset_group in real_dataset_groups:
        for data_set_path in glob.glob(real_dataset_group + '/labeled/train/*'):
            data_set_name = data_set_path.split('/')[-1].replace('.pkl', '')
            dataset = RealPickledDataset(data_set_name, data_set_path)
            datasets.append(dataset)

    for seed in seeds:
        evaluator = Evaluator(datasets, detectors, seed=seed, step=40, sequence_length=)
        evaluator.evaluate()
        result = evaluator.benchmarks()
        evaluator.plot_roc_curves()
        evaluator.plot_threshold_comparison()
        evaluator.plot_scores()
        results = results.append(result, ignore_index=True)

    avg_results = results.groupby(['dataset', 'algorithm'], as_index=False).mean()
    evaluator.set_benchmark_results(avg_results)
    evaluator.export_results('run_real_datasets')
    evaluator.create_boxplots(runs=RUNS, data=results, detectorwise=False)
    evaluator.create_boxplots(runs=RUNS, data=results, detectorwise=True)



if __name__ == '__main__':
    main()

#???????????????????????????????????
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import multivariate_normal
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import trange

from .algorithm_utils import Algorithm, PyTorchUtils


class LSTMED(Algorithm, PyTorchUtils):
    def __init__(self, name: str = 'LSTM-ED', num_epochs: int = 20, batch_size: int = 32, lr: float = 1e-3,
                 hidden_size: int = 5, sequence_length: int = 784, train_gaussian_percentage: float = 0.20,
                 n_layers: tuple = (1, 1), use_bias: tuple = (True, True), dropout: tuple = (0, 0),
                 seed: int = None, gpu: int = None, details=True, step: int = 500):
        Algorithm.__init__(self, __name__, name, seed, details=details)
        PyTorchUtils.__init__(self, seed, gpu)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.step = step
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.train_gaussian_percentage = train_gaussian_percentage

        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout

        self.lstmed = None
        self.proto_input_space_ind = None
        self.hidden_and_prototype_as_df = pd.DataFrame(columns=['hidden_and_prototype_sequences', 'indicator'])
        self.mean, self.cov,self.epoch_loss = None, None, None

    def loss_e(self,prototype,enc_hidden):
        k_list = []
        for k in range(prototype.shape[0]):
            b_list = []
            for batch in range(enc_hidden.shape[0]):
                l = torch.sum(torch.mul(prototype[k]-enc_hidden[batch], prototype[k]-enc_hidden[batch]))
                b_list.append(l)
            b_list = torch.stack(b_list)
            min_b = torch.min(b_list)
            k_list.append(min_b)
        k_list = torch.stack(k_list)
        return torch.sum(k_list)

    def loss_d(self,prototype,d_min = 2.0):
        sum = torch.tensor(0)
        for i in range(prototype.shape[0]):
            for j in range(i+1,prototype.shape[0]):
                sum = sum + torch.square(torch.max(torch.tensor(0),d_min - torch.sqrt_(torch.sum(torch.mul(
                    prototype[i] - prototype[j],prototype[i] - prototype[j]),0))))
        return sum



    def loss_c(self,prototype,enc_hidden):
        b_list = []
        for batch in range(enc_hidden.shape[0]):
            k_list = []
            for k in range(prototype.shape[0]):
                l = torch.sum(torch.mul(prototype[k]-enc_hidden[batch],prototype[k]-enc_hidden[batch]))
                k_list.append(l)
            k_list = torch.stack(k_list)
            min_e = torch.min(k_list)
            b_list.append(min_e)
        b_list = torch.stack(b_list)
        return torch.sum(b_list)


    def fit(self, X: pd.DataFrame):
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        sequences = [data[i:i + self.sequence_length] for i in range(0, data.shape[0] - self.sequence_length +1, self.step)]
        indices = np.random.permutation(len(sequences))
        split_point = int(self.train_gaussian_percentage * len(sequences))
        train_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=True,
                                  sampler=SubsetRandomSampler(indices[:-split_point]), pin_memory=True)
        train_gaussian_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=True,
                                           sampler=SubsetRandomSampler(indices[-split_point:]), pin_memory=True)

        self.lstmed = LSTMEDModule(X.shape[1], self.hidden_size,
                                   self.n_layers, self.use_bias, self.dropout,
                                   seed=self.seed, gpu=self.gpu)
        self.to_device(self.lstmed)
        optimizer = torch.optim.Adam(self.lstmed.parameters(), lr=self.lr)
        self.lstmed.train()
        epoch_loss = []
        for epoch in trange(self.num_epochs):
            logging.debug(f'Epoch {epoch+1}/{self.num_epochs}.')
            loss_arr = []
            for ts_batch in train_loader:
                output, enc_hidden, a = self.lstmed(self.to_var(ts_batch), return_latent =True)
                loss_c = self.loss_c(self.lstmed.prototype_layer.prototype,enc_hidden)
                loss_d = self.loss_d(self.lstmed.prototype_layer.prototype,d_min= 1.0)
                loss_e = self.loss_e(self.lstmed.prototype_layer.prototype,enc_hidden)
                #loss_w = torch.sum(torch.abs(self.lstmed.hidden2output.weight))
                loss = nn.MSELoss(size_average=False)(output, self.to_var(ts_batch.float()))
                total_loss = loss + 0.01*loss_c + loss_d + 0.1*loss_e
                loss_arr.append(total_loss)
                self.lstmed.zero_grad()
                total_loss.backward()
                optimizer.step()
            epoch_loss.append(sum(loss_arr))

            a_min = torch.ones(self.lstmed.prototype_layer.k)
            self.proto_input_space_ind = torch.Tensor(self.lstmed.prototype_layer.k, 2)
            for i in range(len(sequences)):
                output, enc_hidden, a = self.lstmed(self.to_var(
                    torch.Tensor(sequences[i]).expand(1, sequences[i].shape[0], sequences[i].shape[1])),
                                              return_latent=True)
                self.hidden_and_prototype_as_df.loc[len(self.hidden_and_prototype_as_df)] = [enc_hidden[0].detach().numpy().tolist(), 0]
                for k in range(a.shape[1]):
                    if a_min[k] > a[torch.argmin(a[:, k]), k]:
                        a_min[k] = a[torch.argmin(a[:, k]), k]
                        self.proto_input_space_ind[k, :] = torch.Tensor([i, i + self.sequence_length])
            for k in range(self.lstmed.prototype_layer.prototype.shape[0]):
                self.hidden_and_prototype_as_df.loc[len(self.hidden_and_prototype_as_df)] = [self.lstmed.prototype_layer.prototype[k].detach().numpy().tolist(), 1]
            print(self.proto_input_space_ind)
            print(self.hidden_and_prototype_as_df)

        self.lstmed.eval()
        error_vectors = []
        for ts_batch in train_gaussian_loader:
            output = self.lstmed(self.to_var(ts_batch))
            error = nn.L1Loss(reduce=False)(output, self.to_var(ts_batch.float()))
            error_vectors += list(error.view(-1, X.shape[1]).data.cpu().numpy())

        self.mean = np.mean(error_vectors, axis=0)
        self.cov = np.cov(error_vectors, rowvar=False)
        self.epoch_loss = epoch_loss

    def predict(self, X: pd.DataFrame):
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        sequences = [data[i:i + self.sequence_length] for i in range(0,data.shape[0] - self.sequence_length +1,self.step)]
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
            lattice[i % self.sequence_length, i:i + self.sequence_length] = score
        scores = np.nanmean(lattice, axis=0)

        if self.details:
            outputs = np.concatenate(outputs)
            lattice = np.full((self.sequence_length, X.shape[0], X.shape[1]), np.nan)
            for i, output in enumerate(outputs):
                lattice[i % self.sequence_length, i:i + self.sequence_length, :] = output
            self.prediction_details.update({'reconstructions_mean': np.nanmean(lattice, axis=0).T})

            errors = np.concatenate(errors)
            lattice = np.full((self.sequence_length, X.shape[0], X.shape[1]), np.nan)
            for i, error in enumerate(errors):
                lattice[i % self.sequence_length, i:i + self.sequence_length, :] = error
            self.prediction_details.update({'errors_mean': np.nanmean(lattice, axis=0).T})

        return scores

class prototype_layer(nn.Module, PyTorchUtils):
    def __init__(self, hidden_size: int, seed:int, gpu:int, k=5):
        super().__init__()
        PyTorchUtils.__init__(self, seed, gpu)
        self.hidden_size = hidden_size
        self.k = k
        self.prototype_size = torch.Tensor(k, hidden_size)
        self.init_values = nn.init.uniform(self.prototype_size, a=0.0, b=1.0)
        self.prototype = nn.Parameter(self.init_values)
        self.similarity2output = nn.Linear(self.k, 2)

    def forward(self,x,batch_size):
        a = torch.zeros((batch_size,self.k))
        d = x[0].unsqueeze(1) - self.prototype.unsqueeze(1)
        for i in range(self.k):
            a[:,i] = torch.exp(-torch.sum(torch.mul(d[0,i],d[0,i]),1))
        return a






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

        self.encoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
                               num_layers=self.n_layers[0], bias=self.use_bias[0], dropout=self.dropout[0])
        self.to_device(self.encoder)

        self.decoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
                               num_layers=self.n_layers[1], bias=self.use_bias[1], dropout=self.dropout[1])
        self.to_device(self.decoder)
        #-----------
        self.prototype_layer = prototype_layer(self.hidden_size, seed, gpu, k=5)
        self.to_device(self.prototype_layer)
        #-----------
        self.hidden2output = nn.Linear(self.hidden_size, self.n_features)
        self.to_device(self.hidden2output)

    def _init_hidden(self, batch_size):
        return (self.to_var(torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_()),
                self.to_var(torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_()))


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
            output[:, i, :] = self.hidden2output(dec_hidden[0][0, :])

            if self.training:
                _, dec_hidden = self.decoder(ts_batch[:, i].unsqueeze(1).float(), dec_hidden)
            else:
                _, dec_hidden = self.decoder(output[:, i].unsqueeze(1), dec_hidden)

        a = self.prototype_layer(enc_hidden, batch_size)

        return (output, enc_hidden[1][-1], a) if return_latent else output

##################EVALUATOR#######
import gc
import logging
import os
import pickle
import re
import sys
import traceback
from textwrap import wrap
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
import progressbar
import time
from sklearn.metrics import accuracy_score, fbeta_score
from sklearn.metrics import precision_recall_fscore_support as prf
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.manifold._t_sne import TSNE
from tabulate import tabulate

from .config import init_logging


class Evaluator:
    def __init__(self, datasets: list, detectors: callable,step, sequence_length, output_dir: {str} = None, seed: int = None,
                 create_log_file=True):
        """
        :param datasets: list of datasets
        :param detectors: callable that returns list of detectors
        """
        assert np.unique([x.name for x in datasets]).size == len(datasets), 'Some datasets have the same name!'
        self.datasets = datasets
        self._detectors = detectors
        self.step = step
        self.sequence_length = sequence_length
        self.output_dir = output_dir or 'reports'
        self.results = dict()
        self.proto_input_space_ind = dict()
        if create_log_file:
            init_logging(os.path.join(self.output_dir, 'logs'))
        self.logger = logging.getLogger(__name__)
        # Dirty hack: Is set by the main.py to insert results from multiple evaluator runs
        self.benchmark_results = None
        # Last passed seed value in evaluate()
        self.seed = seed

    @property
    def detectors(self):
        detectors = self._detectors(self.seed,self.step,self.sequence_length)
        assert np.unique([x.name for x in detectors]).size == len(detectors), 'Some detectors have the same name!'
        return detectors

    def set_benchmark_results(self, benchmark_result):
        self.benchmark_results = benchmark_result

    def export_results(self, name):
        output_dir = os.path.join(self.output_dir, 'evaluators')
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime('%Y-%m-%d-%H%M%S')
        path = os.path.join(output_dir, f'{name}-{timestamp}.pkl')
        path1 = os.path.join(output_dir, f'{name}-{timestamp}-benchmark_results.csv')
        self.logger.info(f'Store evaluator results at {os.path.abspath(path)}')
        save_dict = {
            'datasets': [x.name for x in self.datasets],
            'detectors': [x.name for x in self.detectors],
            'benchmark_results': self.benchmark_results,
            'results': self.results,
            'output_dir': self.output_dir,
            'seed': int(self.seed),
        }
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
            print(save_dict['benchmark_results'])
        return path

    # Import benchmark_results if this evaluator uses the same detectors and datasets
    # self.results are not available because they are overwritten by each run
    def import_results(self, name):
        output_dir = os.path.join(self.output_dir, 'evaluators')
        path = os.path.join(output_dir, f'{name}.pkl')
        self.logger.info(f'Read evaluator results at {os.path.abspath(path)}')
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)

        self.logger.debug(f'Importing detectors {"; ".join(save_dict["detectors"])}')
        my_detectors = [x.name for x in self.detectors]
        assert np.array_equal(save_dict['detectors'], my_detectors), 'Detectors should be the same'

        self.logger.debug(f'Importing datasets {"; ".join(save_dict["datasets"])}')
        my_datasets = [x.name for x in self.datasets]
        assert np.array_equal(save_dict['datasets'], my_datasets), 'Datasets should be the same'

        self.benchmark_results = save_dict['benchmark_results']
        self.seed = save_dict['seed']
        self.results = save_dict['results']

    @staticmethod
    def get_accuracy_precision_recall_fscore(y_true: list, y_pred: list):
        accuracy = accuracy_score(y_true, y_pred)
        # warn_for=() avoids log warnings for any result being zero
        precision, recall, f_score, _ = prf(y_true, y_pred, average='binary', warn_for=())
        if precision == 0 and recall == 0:
            f01_score = 0
        else:
            f01_score = fbeta_score(y_true, y_pred, average='binary', beta=0.05)
        return accuracy, precision, recall, f_score, f01_score

    @staticmethod
    def get_auroc(det, ds, score):
        if np.isnan(score).all():
            score = np.zeros_like(score)
        _, _, _, y_test = ds.data()
        score_nonan = score.copy()
        # Rank NaN below every other value in terms of anomaly score
        score_nonan[np.isnan(score_nonan)] = np.nanmin(score_nonan) - sys.float_info.epsilon
        fpr, tpr, _ = roc_curve(y_test, score_nonan)
        return auc(fpr, tpr)

    def get_optimal_threshold(self, det, y_test, score, steps=100, return_metrics=False):
        maximum = np.nanmax(score)
        minimum = np.nanmin(score)
        threshold = np.linspace(minimum, maximum, steps)
        metrics = list(self.get_metrics_by_thresholds(y_test, score, threshold))
        metrics = np.array(metrics).T
        anomalies, acc, prec, rec, f_score, f01_score = metrics
        if return_metrics:
            return anomalies, acc, prec, rec, f_score, f01_score, threshold
        else:
            return threshold[np.argmax(f_score)]

    def evaluate(self):
        for ds in progressbar.progressbar(self.datasets):
            (X_train, y_train, X_test, y_test) = ds.data()
            for det in progressbar.progressbar(self.detectors):
                self.logger.info(f'Training {det.name} on {ds.name} with seed {self.seed}')
                try:
                    det.fit(X_train.copy())
                    score = det.predict(X_test.copy())
                    self.results[(ds.name, det.name)] = score
                    self.proto_input_space_ind[(ds.name, det.name)] = det.proto_input_space_ind
                    self.plot_prototypes_in_input_space(det, ds)
                    try:
                        self.plot_latents_and_prototypes(det, ds, det.hidden_and_prototype_as_df)
                        self.plot_details(det, ds, score)
                        self.plot_epoch_loss(det,ds)
                    except Exception:
                        pass
                except Exception as e:
                    self.logger.error(f'An exception occurred while training {det.name} on {ds}: {e}')
                    self.logger.error(traceback.format_exc())
                    self.results[(ds.name, det.name)] = np.zeros_like(y_test)
            gc.collect()

    def benchmarks(self) -> pd.DataFrame:
        df = pd.DataFrame()
        for ds in self.datasets:
            _, _, _, y_test = ds.data()
            for det in self.detectors:
                score = self.results[(ds.name, det.name)]
                y_pred = self.binarize(score, self.get_optimal_threshold(det, y_test, np.array(score)))
                acc, prec, rec, f1_score, f01_score = self.get_accuracy_precision_recall_fscore(y_test, y_pred)
                confusion_mat = confusion_matrix(y_test,y_pred,labels=[0,1])
                self.plot_confusion_matrix(det,ds,confusion_mat)
                score = self.results[(ds.name, det.name)]
                auroc = self.get_auroc(det, ds, score)
                df = df.append({'dataset': ds.name,
                                'algorithm': det.name,
                                'accuracy': acc,
                                'precision': prec,
                                'recall': rec,
                                'F1-score': f1_score,
                                'F0.1-score': f01_score,
                                'auroc': auroc,
                                'confusion_mat':confusion_mat},
                               ignore_index=True)
        return df

    # plot_prototypes_in_input_space plots/ highlights the input sequences which are closer to the prototypes with respect
    # the distance measure 'a= exp(-d)' in the hidden space.
    def plot_prototypes_in_input_space(self,det,ds,store = True):
        plt.close('all')
        fig = plt.figure()
        X_train, y_train, X_test, y_test = ds.data()
        n_proto = self.proto_input_space_ind[(ds.name,det.name)].shape[0]
        for k in range(n_proto):
            st_ind = int(self.proto_input_space_ind[(ds.name,det.name)][k,0].item()) * self.step
            seq_len = self.sequence_length
            end_ind = st_ind + seq_len
            for col in X_train.columns:
                plt.subplot(n_proto,1,k+1)
                plt.plot(X_train[col].iloc[(st_ind-(seq_len*2)):(end_ind+(seq_len*2))])
                plt.plot(X_train[col].iloc[st_ind:end_ind])
                plt.ylabel(f'Proto-{k}')
        if store:
            self.store(fig, f'inputs_closest_to_prototypes-{det.name}-{ds.name}')
        return fig

    # plot_latents_and_prototypes function gathers the hidden space for the inputs along with the prototypes and maps them
    # to 2 dimensions by using tnse approach of dimensionality reduction and plots them as a points in the latent space.
    def plot_latents_and_prototypes(self,det,ds,df,store = True):
        plt.close('all')
        df1 = df.hidden_and_prototype_sequences.apply(pd.Series).assign(**{'indicator':df.indicator})
        tnse = TSNE(n_components = 2 , verbose = 0, perplexity = 5 , n_iter = 300)
        tnse_results = tnse.fit_transform(df1.iloc[:,:-1])
        fig = plt.figure()
        plt.scatter(tnse_results[:,0],tnse_results[:,1],marker='.',c=df1.indicator,cmap='bwr_r')
        if store:
            self.store(fig, f'latents_plot-{det.name}-{ds.name}')
        return fig

    def plot_confusion_matrix(self,det,ds,conf_matrix,store=True):
        plt.close('all')
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

        plt.xlabel('Predictions', fontsize=18)
        plt.ylabel('Actuals', fontsize=18)
        plt.title('Confusion Matrix', fontsize=18)
        if store:
            self.store(fig, f'confusion_matrix-{det.name}-{ds.name}')
        return fig

    def plot_epoch_loss(self,det,ds,store=True):
        plt.close('all')
        fig = plt.figure()
        arr = []
        for i in det.epoch_loss:
            arr.append(i.item())

        plt.plot(arr)
        if store:
            self.store(fig, f'epoch_loss_plot-{det.name}-{ds.name}')
        return fig

    def get_metrics_by_thresholds(self, y_test: list, score: list, thresholds: list):
        for threshold in thresholds:
            anomaly = self.binarize(score, threshold=threshold)
            metrics = Evaluator.get_accuracy_precision_recall_fscore(y_test, anomaly)
            yield (anomaly.sum(), *metrics)

    def plot_scores(self, store=True):
        detectors = self.detectors
        plt.close('all')
        figures = []
        for ds in self.datasets:
            X_train, y_train, X_test, y_test = ds.data()
            subtitle_loc = 'left'
            fig = plt.figure(figsize=(15, 15))
            fig.canvas.set_window_title(ds.name)

            sp = fig.add_subplot((2 * len(detectors) + 3), 1, 1)
            sp.set_title('original training data', loc=subtitle_loc)
            for col in X_train.columns:
                plt.plot(X_train[col], alpha=0.5)
                for det in detectors:
                    for k in range(self.proto_input_space_ind[(ds.name, det.name)].shape[0]):
                        st_ind = int(self.proto_input_space_ind[(ds.name, det.name)][k, 0].item()) * self.step
                        seq_len = self.sequence_length
                        end_ind = st_ind + seq_len
                        for col in X_train.columns:
                            plt.plot(X_train[col].iloc[st_ind:end_ind])
                            plt.axvspan(st_ind, end_ind, color='black')
            sp = fig.add_subplot((2 * len(detectors) + 3), 1, 2)
            sp.set_title('original test set', loc=subtitle_loc)
            for col in X_test.columns:
                plt.plot(X_test[col])

            sp = fig.add_subplot((2 * len(detectors) + 3), 1, 3)
            sp.set_title('binary labels of test set', loc=subtitle_loc)
            plt.plot(y_test)

            subplot_num = 4
            for det in detectors:
                sp = fig.add_subplot((2 * len(detectors) + 3), 1, subplot_num)
                sp.set_title(f'scores of {det.name}', loc=subtitle_loc)
                score = self.results[(ds.name, det.name)]
                plt.plot(np.arange(len(score)), [x for x in score])
                threshold_line = len(score) * [self.get_optimal_threshold(det, y_test, np.array(score))]
                plt.plot([x for x in threshold_line])
                subplot_num += 1

                sp = fig.add_subplot((2 * len(detectors) + 3), 1, subplot_num)
                sp.set_title(f'binary labels of {det.name}', loc=subtitle_loc)
                plt.plot(np.arange(len(score)),
                         [x for x in self.binarize(score, self.get_optimal_threshold(det, y_test, np.array(score)))])
                subplot_num += 1
            fig.subplots_adjust(top=0.9, hspace=0.4)
            fig.tight_layout()
            if store:
                self.store(fig, f'scores_{ds.name}')
            figures.append(fig)
        return figures

    def plot_threshold_comparison(self, steps=40, store=True):
        detectors = self.detectors
        plt.close('all')
        plots_shape = len(detectors), len(self.datasets)
        fig, axes = plt.subplots(*plots_shape, figsize=(len(detectors) * 15, len(self.datasets) * 5))
        # Ensure two dimensions for iteration
        axes = np.array(axes).reshape(*plots_shape).T
        plt.suptitle('Compare thresholds', fontsize=10)
        for ds, axes_row in zip(self.datasets, axes):
            _, _, X_test, y_test = ds.data()

            for det, ax in zip(detectors, axes_row):
                score = np.array(self.results[(ds.name, det.name)])

                anomalies, _, prec, rec, f_score, f01_score, thresh = self.get_optimal_threshold(
                    det, y_test, score, return_metrics=True)

                ax.plot(thresh, anomalies / len(y_test),
                        label=fr'anomalies ({len(y_test)} $\rightarrow$ 1)')
                ax.plot(thresh, prec, label='precision')
                ax.plot(thresh, rec, label='recall')
                ax.plot(thresh, f_score, label='f_score', linestyle='dashed')
                ax.plot(thresh, f01_score, label='f01_score', linestyle='dashed')
                ax.set_title(f'{det.name} on {ds.name}')
                ax.set_xlabel('Threshold')
                ax.legend()

        # Avoid overlapping title and axis labels
        plt.xlim([0.0, 1.0])
        fig.subplots_adjust(top=0.9, hspace=0.4, right=1, left=0)
        fig.tight_layout()
        if store:
            self.store(fig, 'metrics_by_thresholds')
        return fig

    def plot_roc_curves(self, store=True):
        detectors = self.detectors
        plt.close('all')
        figures = []
        for ds in self.datasets:
            _, _, _, y_test = ds.data()
            fig_scale = 3
            fig = plt.figure(figsize=(fig_scale * len(detectors), fig_scale))
            fig.canvas.set_window_title(ds.name + ' ROC')
            fig.suptitle(f'ROC curve on {ds.name}', fontsize=14, y='1.1')
            subplot_count = 1
            for det in detectors:
                self.logger.info(f'Plotting ROC curve for {det.name} on {ds.name}')
                score = self.results[(ds.name, det.name)]
                if np.isnan(score).all():
                    score = np.zeros_like(score)
                # Rank NaN below every other value in terms of anomaly score
                score[np.isnan(score)] = np.nanmin(score) - sys.float_info.epsilon
                fpr, tpr, _ = roc_curve(y_test, score)
                roc_auc = auc(fpr, tpr)
                plt.subplot(1, len(detectors), subplot_count)
                plt.plot(fpr, tpr, color='darkorange',
                         lw=2, label='area = %0.2f' % roc_auc)
                subplot_count += 1
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.gca().set_aspect('equal', adjustable='box')
                plt.title('\n'.join(wrap(det.name, 20)))
                plt.legend(loc='lower right')
            plt.tight_layout()
            if store:
                self.store(fig, f'roc_{ds.name}')
            figures.append(fig)
        return figures

    def plot_auroc(self, store=True, title='AUROC'):
        plt.close('all')
        self.benchmark_results[['dataset', 'algorithm', 'auroc']].pivot(
            index='algorithm', columns='dataset', values='auroc').plot(kind='bar')
        plt.legend(loc=3, framealpha=0.5)
        plt.xticks(rotation=20)
        plt.ylabel('AUC', rotation='horizontal', labelpad=20)
        plt.title(title)
        plt.ylim(ymin=0, ymax=1)
        plt.tight_layout()
        if store:
            self.store(plt.gcf(), 'auroc', store_in_figures=True)

    def plot_details(self, det, ds, score, store=True):
        if not det.details:
            return
        plt.close('all')
        cmap = plt.get_cmap('inferno')
        _, _, X_test, y_test = ds.data()

        grid = 0
        for value in det.prediction_details.values():
            grid += 1 if value.ndim == 1 else value.shape[0]
        grid += X_test.shape[1]  # data
        grid += 1 + 1  # score and gt

        fig, axes = plt.subplots(grid, 1, figsize=(15, 1.5 * grid))

        i = 0
        c = cmap(i / grid)
        axes[i].set_title('test data')
        for col in X_test.values.T:
            axes[i].plot(col, color=c)
            i += 1
        c = cmap(i / grid)

        axes[i].set_title('test gt data')
        axes[i].plot(y_test.values, color=c)
        i += 1
        c = cmap(i / grid)

        axes[i].set_title('scores')
        axes[i].plot(score, color=c)
        i += 1
        c = cmap(i / grid)

        for key, values in det.prediction_details.items():
            axes[i].set_title(key)
            if values.ndim == 1:
                axes[i].plot(values, color=c)
                i += 1
            elif values.ndim == 2:
                for v in values:
                    axes[i].plot(v, color=c)
                    i += 1
            else:
                self.logger.warning('plot_details: not sure what to do')
            c = cmap(i / grid)

        fig.tight_layout()
        if store:
            self.store(fig, f'details_{det.name}_{ds.name}')
        return fig

    # create boxplot diagrams for auc values for each algorithm/dataset per algorithm/dataset
    def create_boxplots(self, runs, data, detectorwise=True, store=True):
        target = 'algorithm' if detectorwise else 'dataset'
        grouped_by = 'dataset' if detectorwise else 'algorithm'
        relevant_results = data[['algorithm', 'dataset', 'auroc']]
        figures = []
        for det_or_ds in (self.detectors if detectorwise else self.datasets):
            relevant_results[relevant_results[target] == det_or_ds.name].boxplot(by=grouped_by, figsize=(15, 15))
            plt.suptitle('')  # boxplot() adds a suptitle
            plt.title(f'AUC grouped by {grouped_by} for {det_or_ds.name} over {runs} runs')
            plt.ylim(ymin=0, ymax=1)
            plt.tight_layout()
            figures.append(plt.gcf())
            if store:
                self.store(plt.gcf(), f'boxplot_auc_for_{det_or_ds.name}_{runs}_runs', store_in_figures=True)
        return figures

    # create bar charts for averaged pipeline results per algorithm/dataset
    def create_bar_charts(self, runs, detectorwise=True, store=True):
        target = 'algorithm' if detectorwise else 'dataset'
        grouped_by = 'dataset' if detectorwise else 'algorithm'
        relevant_results = self.benchmark_results[['algorithm', 'dataset', 'auroc']]
        figures = []
        for det_or_ds in (self.detectors if detectorwise else self.datasets):
            relevant_results[relevant_results[target] == det_or_ds.name].plot(x=grouped_by, kind='bar', figsize=(7, 7))
            plt.suptitle('')  # boxplot() adds a suptitle
            plt.title(f'AUC for {target} {det_or_ds.name} over {runs} runs')
            plt.ylim(ymin=0, ymax=1)
            plt.tight_layout()
            figures.append(plt.gcf())
            if store:
                self.store(plt.gcf(), f'barchart_auc_for_{det_or_ds.name}_{runs}_runs', store_in_figures=True)
        return figures

    def store(self, fig, title, extension='pdf', no_counters=False, store_in_figures=False):
        timestamp = time.strftime('%Y-%m-%d-%H%M%S')
        if store_in_figures:
            output_dir = os.path.join(self.output_dir, 'figures',title)
        else:
            output_dir = os.path.join(self.output_dir, 'figures', f'seed-{self.seed}',title)
        os.makedirs(output_dir, exist_ok=True)
        counters_str = '' if no_counters else f'-{len(self.detectors)}-{len(self.datasets)}'
        path = os.path.join(output_dir, f'{counters_str}--{timestamp}.{extension}')
        fig.savefig(path)
        self.logger.info(f'Stored plot at {path}')

    def store_text(self, content, title, extension='txt'):
        timestamp = int(time.time())
        output_dir = os.path.join(self.output_dir, 'tables', f'seed-{self.seed}')
        path = os.path.join(output_dir, f'{title}-{len(self.detectors)}-{len(self.datasets)}-{timestamp}.{extension}')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(content)
        self.logger.info(f'Stored {extension} file at {path}')

    def print_merged_table_per_dataset(self, results):
        for ds in self.datasets:
            table = tabulate(results[results['dataset'] == ds.name], headers='keys', tablefmt='psql')
            self.logger.info(f'Dataset: {ds.name}\n{table}')

    def gen_merged_latex_per_dataset(self, results, title_suffix=None, store=True):
        title = f'latex_merged{f"_{title_suffix}" if title_suffix else ""}'
        content = ''
        for ds in self.datasets:
            content += f'''{ds.name}:\n\n{tabulate(results[results['dataset'] == ds.name],
                                                   headers='keys', tablefmt='latex')}\n\n'''
        if store:
            self.store_text(content=content, title=title, extension='tex')
        return content

    def print_merged_table_per_algorithm(self, results):
        for det in self.detectors:
            table = tabulate(results[results['algorithm'] == det.name], headers='keys', tablefmt='psql')
            self.logger.info(f'Detector: {det.name}\n{table}')

    def gen_merged_latex_per_algorithm(self, results, title_suffix=None, store=True):
        title = f'latex_merged{f"_{title_suffix}" if title_suffix else ""}'
        content = ''
        for det in self.detectors:
            content += f'''{det.name}:\n\n{tabulate(results[results['algorithm'] == det.name],
                                   headers='keys', tablefmt='latex')}\n\n'''
        if store:
            self.store_text(content=content, title=title, extension='tex')
        return content

    @staticmethod
    def translate_var_key(key_name):
        if key_name == 'pol':
            return 'Pollution'
        if key_name == 'mis':
            return 'Missing'
        if key_name == 'extremeness':
            return 'Extremeness'
        if key_name == 'f':
            return 'Multivariate'
        # self.logger('Unexpected dataset name (unknown variable in name)')
        return None

    @staticmethod
    def get_key_and_value(dataset_name):
        # Extract var name and value from dataset name
        var_re = re.compile(r'.+\((\w*)=(.*)\)')
        # e.g. 'Syn Extreme Outliers (pol=0.1)'
        match = var_re.search(dataset_name)
        if not match:
            # self.logger.warn('Unexpected dataset name (not variable in name)')
            return '-', dataset_name
        var_key = match.group(1)
        var_value = match.group(2)
        return Evaluator.translate_var_key(var_key), var_value

    @staticmethod
    def get_dataset_types(mi_df):
        types = mi_df.index.get_level_values('Type')
        indexes = np.unique(types, return_index=True)[1]
        return [types[index] for index in sorted(indexes)]

    @staticmethod
    def insert_multi_index_yaxis(ax, mi_df):
        type_title_offset = -1.6  # depends on string length of xaxis ticklabels

        datasets = mi_df.index
        dataset_types = Evaluator.get_dataset_types(mi_df)  # Returns unique entries keeping original order
        logging.getLogger(__name__).debug('Plotting heatmap for groups {" ".join(dataset_types)}')

        ax.set_yticks(np.arange(len(datasets)))
        ax.set_yticklabels([x[1] for x in datasets])

        y_axis_title_pos = 0  # Store at which position we are for plotting the next title
        for idx, dataset_type in enumerate(dataset_types):
            section_frame = mi_df.iloc[mi_df.index.get_level_values('Type') == dataset_type]
            # Somehow it's sorted by its occurence (which is what we want here)
            dataset_levels = section_frame.index.remove_unused_levels().levels[1]
            title_pos = y_axis_title_pos + 0.5 * (len(dataset_levels) - 1)
            ax.text(type_title_offset, title_pos, dataset_type, ha='center', va='center', rotation=90,
                    fontproperties=FontProperties(weight='bold'))
            if idx < len(dataset_types) - 1:
                sep_pos = y_axis_title_pos + (len(dataset_levels) - 0.6)
                ax.text(-0.5, sep_pos, '_' * int(type_title_offset * -10), ha='right', va='center')
            y_axis_title_pos += len(dataset_levels)

    @staticmethod
    def to_multi_index_frame(evaluators):
        evaluator = evaluators[0]
        for other_evaluator in evaluators[1:]:
            assert evaluator.detectors == other_evaluator.detectors, 'All evaluators should use the same detectors'
        pivot_benchmarks = [ev.benchmark_results.pivot(index='dataset', columns='algorithm',
                                                       values='auroc') for ev in evaluators]

        concat_benchmarks = pd.concat(pivot_benchmarks)
        auroc_matrix = concat_benchmarks.groupby(['dataset']).mean()

        datasets = [[evaluator.get_key_and_value(str(d)) for d in ev.index.values]
                    for ev in pivot_benchmarks]
        datasets = [tuple(d) for d in np.concatenate(datasets)]  # Required for MultiIndex.from_tuples
        datasets = pd.MultiIndex.from_tuples(datasets, names=['Type', 'Level'])
        auroc_matrix.index = datasets
        return auroc_matrix

    def get_multi_index_dataframe(self):
        return Evaluator.to_multi_index_frame([self])

    @staticmethod
    def plot_heatmap(evaluators, store=True):
        mi_df = Evaluator.to_multi_index_frame(evaluators)
        detectors, datasets = mi_df.columns, mi_df.index

        fig, ax = plt.subplots(figsize=(len(detectors) + 2, len(datasets)))
        im = ax.imshow(mi_df, cmap=plt.get_cmap('YlOrRd'), vmin=0, vmax=1)
        plt.colorbar(im)

        # Show MultiIndex for ordinate
        Evaluator.insert_multi_index_yaxis(ax, mi_df)

        # Rotate the tick labels and set their alignment.
        ax.set_xticks(np.arange(len(detectors)))
        ax.set_xticklabels(detectors)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

        # Loop over data dimensions and create text annotations.
        for i in range(len(detectors)):
            for j in range(len(datasets)):
                ax.text(i, j, f'{mi_df.iloc[j, i]:.2f}', ha='center', va='center', color='w',
                        path_effects=[path_effects.withSimplePatchShadow(
                            offset=(1, -1), shadow_rgbFace='b', alpha=0.9)])

        ax.set_title('AUROC over all datasets and detectors')
        # Prevent bug where x axis ticks are completely outside of bounds (matplotlib/issues/5456)
        if len(datasets) > 2:
            fig.tight_layout()
        if store:
            evaluators[0].store(fig, 'heatmap', no_counters=True, store_in_figures=True)
        return fig

    def plot_single_heatmap(self, store=True):
        Evaluator.plot_heatmap([self], store)

    @staticmethod
    def get_printable_runs_results(results):
        print_order = ['dataset', 'algorithm', 'accuracy', 'precision', 'recall', 'F1-score', 'F0.1-score', 'auroc']
        rename_columns = [col for col in print_order if col not in ['dataset', 'algorithm']]

        # calc std and mean for each algorithm per dataset
        std_results = results.groupby(['dataset', 'algorithm']).std(ddof=0).fillna(0)
        # get rid of multi-index
        std_results = std_results.reset_index()
        std_results = std_results[print_order]
        std_results.rename(inplace=True, index=str,
                           columns=dict([(old_col, old_col + '_std') for old_col in rename_columns]))

        avg_results = results.groupby(['dataset', 'algorithm'], as_index=False).mean()
        avg_results = avg_results[print_order]

        avg_results_renamed = avg_results.rename(
            index=str, columns=dict([(old_col, old_col + '_avg') for old_col in rename_columns]))
        return std_results, avg_results, avg_results_renamed

    def gen_merged_tables(self, results, title_suffix=None, store=True):
        title_suffix = f'_{title_suffix}' if title_suffix else ''
        std_results, avg_results, avg_results_renamed = Evaluator.get_printable_runs_results(results)

        ds_title_suffix = f'per_dataset{title_suffix}'
        self.print_merged_table_per_dataset(std_results)
        self.gen_merged_latex_per_dataset(std_results, f'std_{ds_title_suffix}', store=store)

        self.print_merged_table_per_dataset(avg_results_renamed)
        self.gen_merged_latex_per_dataset(avg_results_renamed, f'avg_{ds_title_suffix}', store=store)

        det_title_suffix = f'per_algorithm{title_suffix}'
        self.print_merged_table_per_algorithm(std_results)

        self.gen_merged_latex_per_algorithm(std_results, f'std_{det_title_suffix}', store=store)
        self.print_merged_table_per_algorithm(avg_results_renamed)
        self.gen_merged_latex_per_algorithm(avg_results_renamed, f'avg_{det_title_suffix}', store=store)

    def binarize(self, score, threshold=None):
        threshold = threshold if threshold is not None else self.threshold(score)
        score = np.where(np.isnan(score), np.nanmin(score) - sys.float_info.epsilon, score)
        return np.where(score >= threshold, 1, 0)

    def threshold(self, score):
        return np.nanmean(score) + 2 * np.nanstd(score)


####################################################

import os
import pickle

import numpy as np
import pandas as pd
from sklearn import preprocessing

from .dataset import Dataset


class RealDataset(Dataset):
    def __init__(self, raw_path, **kwargs):
        super().__init__(**kwargs)
        self.raw_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/raw/", raw_path)


class RealPickledDataset(Dataset):
    """Class for pickled datasets from https://github.com/chickenbestlover/RNN-Time-series-Anomaly-Detection"""

    def __init__(self, name, training_path):
        self.name = name
        self.training_path = training_path
        self.test_path = self.training_path.replace("train", "test")
        self._data = None

    def data(self):
        if self._data is None:
            with open(self.training_path, 'rb') as f:
                X_train = pd.DataFrame(pickle.load(f))
                X_train = X_train.iloc[:, :-1]

            #MinMax Normalization
            scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
            names = X_train.columns
            d = scaler.fit_transform(X_train)
            X_train = pd.DataFrame(d, columns=names)

            #Z-score normalization
            #mean, std = X_train.mean(), X_train.std()
            # X_train = (X_train - mean) / std

            with open(self.test_path, 'rb') as f:
                X_test = pd.DataFrame(pickle.load(f))
            y_test = X_test.iloc[:, -1]
            X_test = X_test.iloc[:, :-1]
            #X_test = (X_test - mean) / std

            #Min-Max norm
            scaler1 = preprocessing.MinMaxScaler(feature_range=(-1, 1))
            names1 = X_test.columns
            d1 = scaler1.fit_transform(X_train)
            X_test = pd.DataFrame(d1, columns=names1)



            self._data = X_train, np.zeros(len(X_train)), X_test, y_test
        return self._data

##############################################################################

   def plot_scores(self, store=True):
        detectors = self.detectors
        plt.close('all')
        figures = []
        for ds in self.datasets:
            X_train, y_train, X_test, y_test = ds.data()
            subtitle_loc = 'left'
            fig = plt.figure(figsize=(15, 15))
            fig.canvas.set_window_title(ds.name)

            sp = fig.add_subplot((2 * len(detectors) + 3), 1, 1)
            sp.set_title('original training data', loc=subtitle_loc)
            for col in X_train.columns:
                plt.plot(X_train[col], alpha=0.5)
                for det in detectors:
                    for k in range(self.proto_input_space_ind[(ds.name, det.name)].shape[0]):
                        st_ind = int(self.proto_input_space_ind[(ds.name, det.name)][k, 0].item()) * self.window
                        seq_len = self.sequence_length
                        end_ind = st_ind + seq_len
                        for col in X_train.columns:
                            plt.plot(X_train[col].iloc[st_ind:end_ind])
                            plt.axvspan(st_ind, end_ind, color='black')
            sp = fig.add_subplot((2 * len(detectors) + 3), 1, 2)
            sp.set_title('original test set', loc=subtitle_loc)
            for col in X_test.columns:
                plt.plot(X_test[col])

            sp = fig.add_subplot((2 * len(detectors) + 3), 1, 3)
            sp.set_title('binary labels of test set', loc=subtitle_loc)
            plt.plot(y_test)

            subplot_num = 4
            for det in detectors:
                sp = fig.add_subplot((2 * len(detectors) + 3), 1, subplot_num)
                sp.set_title(f'scores of {det.name}', loc=subtitle_loc)
                score = self.results[(ds.name, det.name)]
                plt.plot(np.arange(len(score)), [x for x in score])
                threshold_line = len(score) * [self.get_optimal_threshold(det, y_test, np.array(score))]
                plt.plot([x for x in threshold_line])
                subplot_num += 1

                sp = fig.add_subplot((2 * len(detectors) + 3), 1, subplot_num)
                sp.set_title(f'binary labels of {det.name}', loc=subtitle_loc)
                plt.plot(np.arange(len(score)),
                         [x for x in self.binarize(score, self.get_optimal_threshold(det, y_test, np.array(score)))])
                subplot_num += 1
            fig.subplots_adjust(top=0.9, hspace=0.4)
            fig.tight_layout()
            if store:
                self.store(fig, f'scores_{ds.name}')
            figures.append(fig)
        return figures

#########################################################################################################################
import glob
import os

import numpy as np
import pandas as pd

from src.algorithms import LSTMED
from src.datasets import RealPickledDataset
from src.evaluation import Evaluator
from tslearn.datasets import UCR_UEA_datasets


RUNS = 1


def main():
    evaluate_real_datasets()


def detectors(seed,window,sequence_length):
    standard_epochs = 1
    dets = [
            LSTMED(num_epochs=standard_epochs, seed=seed, window=window, sequence_length=sequence_length)
            ]

    return sorted(dets, key=lambda x: x.framework)


def evaluate_real_datasets():
    REAL_DATASET_GROUP_PATH = 'data/raw/'
    real_dataset_groups = glob.glob(REAL_DATASET_GROUP_PATH + '*')
    seeds = [3424441233]
    results = pd.DataFrame()
    datasets = []
    for real_dataset_group in real_dataset_groups:
        for data_set_path in glob.glob(real_dataset_group + '/labeled/train/*'):
            data_set_name = data_set_path.split('/')[-1].replace('.pkl', '')
            dataset = RealPickledDataset(data_set_name, data_set_path)
            datasets.append(dataset)

    for seed in seeds:
        evaluator = Evaluator(datasets, detectors, seed=seed, window=1, sequence_length=30)
        evaluator.evaluate()
        result = evaluator.benchmarks()
        evaluator.plot_roc_curves()
        evaluator.plot_threshold_comparison()
        evaluator.plot_scores()
        results = results.append(result, ignore_index=True)

    avg_results = results.groupby(['dataset', 'algorithm'], as_index=False).mean()
    evaluator.set_benchmark_results(avg_results)
    evaluator.export_results('run_real_datasets')
    evaluator.create_boxplots(runs=RUNS, data=results, detectorwise=False)
    evaluator.create_boxplots(runs=RUNS, data=results, detectorwise=True)



if __name__ == '__main__':
    main()

########################################################################################################################

import gc
import logging
import os
import pickle
import re
import sys
import traceback
from textwrap import wrap
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
import progressbar
import time
from sklearn.metrics import accuracy_score, fbeta_score
from sklearn.metrics import precision_recall_fscore_support as prf
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.manifold._t_sne import TSNE
from tabulate import tabulate

from .config import init_logging


class Evaluator:
    def __init__(self, datasets: list, detectors: callable,step, sequence_length, output_dir: {str} = None, seed: int = None,
                 create_log_file=True):
        """
        :param datasets: list of datasets
        :param detectors: callable that returns list of detectors
        """
        assert np.unique([x.name for x in datasets]).size == len(datasets), 'Some datasets have the same name!'
        self.datasets = datasets
        self._detectors = detectors
        self.step = step
        self.sequence_length = sequence_length
        self.output_dir = output_dir or 'reports'
        self.results = dict()
        self.proto_input_space_ind = dict()
        if create_log_file:
            init_logging(os.path.join(self.output_dir, 'logs'))
        self.logger = logging.getLogger(__name__)
        # Dirty hack: Is set by the main.py to insert results from multiple evaluator runs
        self.benchmark_results = None
        # Last passed seed value in evaluate()
        self.seed = seed

    @property
    def detectors(self):
        detectors = self._detectors(self.seed,self.step,self.sequence_length)
        assert np.unique([x.name for x in detectors]).size == len(detectors), 'Some detectors have the same name!'
        return detectors

    def set_benchmark_results(self, benchmark_result):
        self.benchmark_results = benchmark_result

    def export_results(self, name):
        output_dir = os.path.join(self.output_dir, 'evaluators')
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime('%Y-%m-%d-%H%M%S')
        path = os.path.join(output_dir, f'{name}-{timestamp}.pkl')
        path1 = os.path.join(output_dir, f'{name}-{timestamp}-benchmark_results.csv')
        self.logger.info(f'Store evaluator results at {os.path.abspath(path)}')
        save_dict = {
            'datasets': [x.name for x in self.datasets],
            'detectors': [x.name for x in self.detectors],
            'benchmark_results': self.benchmark_results,
            'results': self.results,
            'output_dir': self.output_dir,
            'seed': int(self.seed),
        }
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
            print(save_dict['benchmark_results'])
        return path

    # Import benchmark_results if this evaluator uses the same detectors and datasets
    # self.results are not available because they are overwritten by each run
    def import_results(self, name):
        output_dir = os.path.join(self.output_dir, 'evaluators')
        path = os.path.join(output_dir, f'{name}.pkl')
        self.logger.info(f'Read evaluator results at {os.path.abspath(path)}')
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)

        self.logger.debug(f'Importing detectors {"; ".join(save_dict["detectors"])}')
        my_detectors = [x.name for x in self.detectors]
        assert np.array_equal(save_dict['detectors'], my_detectors), 'Detectors should be the same'

        self.logger.debug(f'Importing datasets {"; ".join(save_dict["datasets"])}')
        my_datasets = [x.name for x in self.datasets]
        assert np.array_equal(save_dict['datasets'], my_datasets), 'Datasets should be the same'

        self.benchmark_results = save_dict['benchmark_results']
        self.seed = save_dict['seed']
        self.results = save_dict['results']

    @staticmethod
    def get_accuracy_precision_recall_fscore(y_true: list, y_pred: list):
        accuracy = accuracy_score(y_true, y_pred)
        # warn_for=() avoids log warnings for any result being zero
        precision, recall, f_score, _ = prf(y_true, y_pred, average='binary', warn_for=())
        if precision == 0 and recall == 0:
            f01_score = 0
        else:
            f01_score = fbeta_score(y_true, y_pred, average='binary', beta=0.05)
        return accuracy, precision, recall, f_score, f01_score

    @staticmethod
    def get_auroc(det, ds, score):
        if np.isnan(score).all():
            score = np.zeros_like(score)
        _, _, _, y_test = ds.data()
        score_nonan = score.copy()
        # Rank NaN below every other value in terms of anomaly score
        score_nonan[np.isnan(score_nonan)] = np.nanmin(score_nonan) - sys.float_info.epsilon
        fpr, tpr, _ = roc_curve(y_test, score_nonan)
        return auc(fpr, tpr)

    def get_optimal_threshold(self, det, y_test, score, steps=100, return_metrics=False):
        maximum = np.nanmax(score)
        minimum = np.nanmin(score)
        threshold = np.linspace(minimum, maximum, steps)
        metrics = list(self.get_metrics_by_thresholds(y_test, score, threshold))
        metrics = np.array(metrics).T
        anomalies, acc, prec, rec, f_score, f01_score = metrics
        if return_metrics:
            return anomalies, acc, prec, rec, f_score, f01_score, threshold
        else:
            return threshold[np.argmax(f_score)]

    def evaluate(self):
        for ds in progressbar.progressbar(self.datasets):
            (X_train, y_train, X_test, y_test) = ds.data()
            for det in progressbar.progressbar(self.detectors):
                self.logger.info(f'Training {det.name} on {ds.name} with seed {self.seed}')
                try:
                    det.fit(X_train.copy())
                    score = det.predict(X_test.copy())
                    self.results[(ds.name, det.name)] = score
                    self.proto_input_space_ind[(ds.name,det.name)] = det.proto_input_space_ind
                    self.plot_prototypes_in_input_space(det,ds)
                    try:
                        self.plot_latents_and_prototypes(det,ds,det.hidden_and_prototype_as_df)
                        self.plot_details(det, ds, score)
                        self.plot_epoch_loss(det,ds)
                    except Exception:
                        pass
                except Exception as e:
                    self.logger.error(f'An exception occurred while training {det.name} on {ds}: {e}')
                    self.logger.error(traceback.format_exc())
                    self.results[(ds.name, det.name)] = np.zeros_like(y_test)
            gc.collect()

    def benchmarks(self) -> pd.DataFrame:
        df = pd.DataFrame()
        for ds in self.datasets:
            _, _, _, y_test = ds.data()
            for det in self.detectors:
                score = self.results[(ds.name, det.name)]
                y_pred = self.binarize(score, self.get_optimal_threshold(det, y_test, np.array(score)))
                acc, prec, rec, f1_score, f01_score = self.get_accuracy_precision_recall_fscore(y_test, y_pred)
                confusion_mat = confusion_matrix(y_test,y_pred,labels=[0,1])
                self.plot_confusion_matrix(det,ds,confusion_mat)
                score = self.results[(ds.name, det.name)]
                auroc = self.get_auroc(det, ds, score)
                df = df.append({'dataset': ds.name,
                                'algorithm': det.name,
                                'accuracy': acc,
                                'precision': prec,
                                'recall': rec,
                                'F1-score': f1_score,
                                'F0.1-score': f01_score,
                                'auroc': auroc,
                                'confusion_mat':confusion_mat
                                },
                               ignore_index=True)
        return df

    # plot_prototypes_in_input_space plots/ highlights the input sequences which are closer to the prototypes with respect
    # the distance measure 'a= exp(-d)' in the hidden space.
    def plot_prototypes_in_input_space(self,det,ds,store = True):
        plt.close('all')
        fig = plt.figure()
        X_train, y_train, X_test, y_test = ds.data()
        n_proto = self.proto_input_space_ind[(ds.name,det.name)].shape[0]
        for k in range(n_proto):
            st_ind = int(self.proto_input_space_ind[(ds.name,det.name)][k,0].item()) * self.step
            seq_len = self.sequence_length
            end_ind = st_ind + seq_len
            for col in X_train.columns:
                plt.subplot(n_proto,1,k+1)
                plt.plot(X_train[col].iloc[(st_ind-(seq_len*2)):(end_ind+(seq_len*2))])
                plt.plot(X_train[col].iloc[st_ind:end_ind])
                plt.ylabel(f'Proto-{k}')
        if store:
            self.store(fig, f'inputs_closest_to_prototypes-{det.name}-{ds.name}')
        return fig

    # plot_latents_and_prototypes function gathers the hidden space for the inputs along with the prototypes and maps them
    # to 2 dimensions by using tnse approach of dimensionality reduction and plots them as a points in the latent space.
    def plot_latents_and_prototypes(self,det,ds,df,store = True):
        plt.close('all')
        df1 = df.hidden_and_prototype_sequences.apply(pd.Series).assign(**{'indicator':df.indicator})
        tnse = TSNE(n_components = 2 , verbose = 0, perplexity = 5 , n_iter = 300)
        tnse_results = tnse.fit_transform(df1.iloc[:,:-1])
        fig = plt.figure()
        plt.scatter(tnse_results[:,0],tnse_results[:,1],marker='.',c=df1.indicator,cmap='bwr_r')
        if store:
            self.store(fig, f'latents_plot-{det.name}-{ds.name}')
        return fig

    def plot_confusion_matrix(self,det,ds,conf_matrix,store=True):
        plt.close('all')
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

        plt.xlabel('Predictions', fontsize=18)
        plt.ylabel('Actuals', fontsize=18)
        plt.title('Confusion Matrix', fontsize=18)
        if store:
            self.store(fig, f'confusion_matrix-{det.name}-{ds.name}')
        return fig

    def plot_epoch_loss(self,det,ds,store=True):
        plt.close('all')
        fig = plt.figure()
        arr = []
        for i in det.epoch_loss:
            arr.append(i.item())

        plt.plot(arr)
        if store:
            self.store(fig, f'epoch_loss_plot-{det.name}-{ds.name}')
        return fig

    def get_metrics_by_thresholds(self, y_test: list, score: list, thresholds: list):
        for threshold in thresholds:
            anomaly = self.binarize(score, threshold=threshold)
            metrics = Evaluator.get_accuracy_precision_recall_fscore(y_test, anomaly)
            yield (anomaly.sum(), *metrics)

    def plot_scores(self, store=True):
        detectors = self.detectors
        plt.close('all')
        figures = []
        for ds in self.datasets:
            X_train, y_train, X_test, y_test = ds.data()
            subtitle_loc = 'left'
            fig = plt.figure(figsize=(15, 15))
            fig.canvas.set_window_title(ds.name)

            sp = fig.add_subplot((2 * len(detectors) + 3), 1, 1)
            sp.set_title('original training data', loc=subtitle_loc)
            for col in X_train.columns:
                plt.plot(X_train[col], alpha=0.5)
            for det in detectors:
                for k in range(self.proto_input_space_ind[(ds.name, det.name)].shape[0]):
                        st_ind = int(self.proto_input_space_ind[(ds.name, det.name)][k, 0].item()) * self.step
                        seq_len = self.sequence_length
                        end_ind = st_ind + seq_len
                        for col in X_train.columns:
                            plt.plot(X_train[col].iloc[st_ind:end_ind])
                            plt.axvspan(st_ind, end_ind, color='black')
            sp = fig.add_subplot((2 * len(detectors) + 3), 1, 2)
            sp.set_title('original test set', loc=subtitle_loc)
            for col in X_test.columns:
                plt.plot(X_test[col])

            sp = fig.add_subplot((2 * len(detectors) + 3), 1, 3)
            sp.set_title('binary labels of test set', loc=subtitle_loc)
            plt.plot(y_test)

            subplot_num = 4
            for det in detectors:
                sp = fig.add_subplot((2 * len(detectors) + 3), 1, subplot_num)
                sp.set_title(f'scores of {det.name}', loc=subtitle_loc)
                score = self.results[(ds.name, det.name)]
                plt.plot(np.arange(len(score)), [x for x in score])
                threshold_line = len(score) * [self.get_optimal_threshold(det, y_test, np.array(score))]
                plt.plot([x for x in threshold_line])
                subplot_num += 1

                sp = fig.add_subplot((2 * len(detectors) + 3), 1, subplot_num)
                sp.set_title(f'binary labels of {det.name}', loc=subtitle_loc)
                plt.plot(np.arange(len(score)),
                         [x for x in self.binarize(score, self.get_optimal_threshold(det, y_test, np.array(score)))])
                subplot_num += 1
            fig.subplots_adjust(top=0.9, hspace=0.4)
            fig.tight_layout()
            if store:
                self.store(fig, f'scores_{ds.name}')
            figures.append(fig)
        return figures

    def plot_threshold_comparison(self, steps=40, store=True):
        detectors = self.detectors
        plt.close('all')
        plots_shape = len(detectors), len(self.datasets)
        fig, axes = plt.subplots(*plots_shape, figsize=(len(detectors) * 15, len(self.datasets) * 5))
        # Ensure two dimensions for iteration
        axes = np.array(axes).reshape(*plots_shape).T
        plt.suptitle('Compare thresholds', fontsize=10)
        for ds, axes_row in zip(self.datasets, axes):
            _, _, X_test, y_test = ds.data()

            for det, ax in zip(detectors, axes_row):
                score = np.array(self.results[(ds.name, det.name)])

                anomalies, _, prec, rec, f_score, f01_score, thresh = self.get_optimal_threshold(
                    det, y_test, score, return_metrics=True)

                ax.plot(thresh, anomalies / len(y_test),
                        label=fr'anomalies ({len(y_test)} $\rightarrow$ 1)')
                ax.plot(thresh, prec, label='precision')
                ax.plot(thresh, rec, label='recall')
                ax.plot(thresh, f_score, label='f_score', linestyle='dashed')
                ax.plot(thresh, f01_score, label='f01_score', linestyle='dashed')
                ax.set_title(f'{det.name} on {ds.name}')
                ax.set_xlabel('Threshold')
                ax.legend()

        # Avoid overlapping title and axis labels
        plt.xlim([0.0, 1.0])
        fig.subplots_adjust(top=0.9, hspace=0.4, right=1, left=0)
        fig.tight_layout()
        if store:
            self.store(fig, 'metrics_by_thresholds')
        return fig

    def plot_roc_curves(self, store=True):
        detectors = self.detectors
        plt.close('all')
        figures = []
        for ds in self.datasets:
            _, _, _, y_test = ds.data()
            fig_scale = 3
            fig = plt.figure(figsize=(fig_scale * len(detectors), fig_scale))
            fig.canvas.set_window_title(ds.name + ' ROC')
            fig.suptitle(f'ROC curve on {ds.name}', fontsize=14, y='1.1')
            subplot_count = 1
            for det in detectors:
                self.logger.info(f'Plotting ROC curve for {det.name} on {ds.name}')
                score = self.results[(ds.name, det.name)]
                if np.isnan(score).all():
                    score = np.zeros_like(score)
                # Rank NaN below every other value in terms of anomaly score
                score[np.isnan(score)] = np.nanmin(score) - sys.float_info.epsilon
                fpr, tpr, _ = roc_curve(y_test, score)
                roc_auc = auc(fpr, tpr)
                plt.subplot(1, len(detectors), subplot_count)
                plt.plot(fpr, tpr, color='darkorange',
                         lw=2, label='area = %0.2f' % roc_auc)
                subplot_count += 1
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.gca().set_aspect('equal', adjustable='box')
                plt.title('\n'.join(wrap(det.name, 20)))
                plt.legend(loc='lower right')
            plt.tight_layout()
            if store:
                self.store(fig, f'roc_{ds.name}')
            figures.append(fig)
        return figures

    def plot_auroc(self, store=True, title='AUROC'):
        plt.close('all')
        self.benchmark_results[['dataset', 'algorithm', 'auroc']].pivot(
            index='algorithm', columns='dataset', values='auroc').plot(kind='bar')
        plt.legend(loc=3, framealpha=0.5)
        plt.xticks(rotation=20)
        plt.ylabel('AUC', rotation='horizontal', labelpad=20)
        plt.title(title)
        plt.ylim(ymin=0, ymax=1)
        plt.tight_layout()
        if store:
            self.store(plt.gcf(), 'auroc', store_in_figures=True)

    def plot_details(self, det, ds, score, store=True):
        if not det.details:
            return
        plt.close('all')
        cmap = plt.get_cmap('inferno')
        _, _, X_test, y_test = ds.data()

        grid = 0
        for value in det.prediction_details.values():
            grid += 1 if value.ndim == 1 else value.shape[0]
        grid += X_test.shape[1]  # data
        grid += 1 + 1  # score and gt

        fig, axes = plt.subplots(grid, 1, figsize=(15, 1.5 * grid))

        i = 0
        c = cmap(i / grid)
        axes[i].set_title('test data')
        for col in X_test.values.T:
            axes[i].plot(col, color=c)
            i += 1
        c = cmap(i / grid)

        axes[i].set_title('test gt data')
        axes[i].plot(y_test.values, color=c)
        i += 1
        c = cmap(i / grid)

        axes[i].set_title('scores')
        axes[i].plot(score, color=c)
        i += 1
        c = cmap(i / grid)

        for key, values in det.prediction_details.items():
            axes[i].set_title(key)
            if values.ndim == 1:
                axes[i].plot(values, color=c)
                i += 1
            elif values.ndim == 2:
                for v in values:
                    axes[i].plot(v, color=c)
                    i += 1
            else:
                self.logger.warning('plot_details: not sure what to do')
            c = cmap(i / grid)

        fig.tight_layout()
        if store:
            self.store(fig, f'details_{det.name}_{ds.name}')
        return fig

    # create boxplot diagrams for auc values for each algorithm/dataset per algorithm/dataset
    def create_boxplots(self, runs, data, detectorwise=True, store=True):
        target = 'algorithm' if detectorwise else 'dataset'
        grouped_by = 'dataset' if detectorwise else 'algorithm'
        relevant_results = data[['algorithm', 'dataset', 'auroc']]
        figures = []
        for det_or_ds in (self.detectors if detectorwise else self.datasets):
            relevant_results[relevant_results[target] == det_or_ds.name].boxplot(by=grouped_by, figsize=(15, 15))
            plt.suptitle('')  # boxplot() adds a suptitle
            plt.title(f'AUC grouped by {grouped_by} for {det_or_ds.name} over {runs} runs')
            plt.ylim(ymin=0, ymax=1)
            plt.tight_layout()
            figures.append(plt.gcf())
            if store:
                self.store(plt.gcf(), f'boxplot_auc_for_{det_or_ds.name}_{runs}_runs', store_in_figures=True)
        return figures

    # create bar charts for averaged pipeline results per algorithm/dataset
    def create_bar_charts(self, runs, detectorwise=True, store=True):
        target = 'algorithm' if detectorwise else 'dataset'
        grouped_by = 'dataset' if detectorwise else 'algorithm'
        relevant_results = self.benchmark_results[['algorithm', 'dataset', 'auroc']]
        figures = []
        for det_or_ds in (self.detectors if detectorwise else self.datasets):
            relevant_results[relevant_results[target] == det_or_ds.name].plot(x=grouped_by, kind='bar', figsize=(7, 7))
            plt.suptitle('')  # boxplot() adds a suptitle
            plt.title(f'AUC for {target} {det_or_ds.name} over {runs} runs')
            plt.ylim(ymin=0, ymax=1)
            plt.tight_layout()
            figures.append(plt.gcf())
            if store:
                self.store(plt.gcf(), f'barchart_auc_for_{det_or_ds.name}_{runs}_runs', store_in_figures=True)
        return figures

    def store(self, fig, title, extension='pdf', no_counters=False, store_in_figures=False):
        timestamp = time.strftime('%Y-%m-%d-%H%M%S')
        if store_in_figures:
            output_dir = os.path.join(self.output_dir, 'figures',title)
        else:
            output_dir = os.path.join(self.output_dir, 'figures', f'seed-{self.seed}',title)
        os.makedirs(output_dir, exist_ok=True)
        counters_str = '' if no_counters else f'-{len(self.detectors)}-{len(self.datasets)}'
        path = os.path.join(output_dir, f'{counters_str}--{timestamp}.{extension}')
        fig.savefig(path)
        self.logger.info(f'Stored plot at {path}')

    def store_text(self, content, title, extension='txt'):
        timestamp = int(time.time())
        output_dir = os.path.join(self.output_dir, 'tables', f'seed-{self.seed}')
        path = os.path.join(output_dir, f'{title}-{len(self.detectors)}-{len(self.datasets)}-{timestamp}.{extension}')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(content)
        self.logger.info(f'Stored {extension} file at {path}')

    def print_merged_table_per_dataset(self, results):
        for ds in self.datasets:
            table = tabulate(results[results['dataset'] == ds.name], headers='keys', tablefmt='psql')
            self.logger.info(f'Dataset: {ds.name}\n{table}')

    def gen_merged_latex_per_dataset(self, results, title_suffix=None, store=True):
        title = f'latex_merged{f"_{title_suffix}" if title_suffix else ""}'
        content = ''
        for ds in self.datasets:
            content += f'''{ds.name}:\n\n{tabulate(results[results['dataset'] == ds.name],
                                                   headers='keys', tablefmt='latex')}\n\n'''
        if store:
            self.store_text(content=content, title=title, extension='tex')
        return content

    def print_merged_table_per_algorithm(self, results):
        for det in self.detectors:
            table = tabulate(results[results['algorithm'] == det.name], headers='keys', tablefmt='psql')
            self.logger.info(f'Detector: {det.name}\n{table}')

    def gen_merged_latex_per_algorithm(self, results, title_suffix=None, store=True):
        title = f'latex_merged{f"_{title_suffix}" if title_suffix else ""}'
        content = ''
        for det in self.detectors:
            content += f'''{det.name}:\n\n{tabulate(results[results['algorithm'] == det.name],
                                   headers='keys', tablefmt='latex')}\n\n'''
        if store:
            self.store_text(content=content, title=title, extension='tex')
        return content

    @staticmethod
    def translate_var_key(key_name):
        if key_name == 'pol':
            return 'Pollution'
        if key_name == 'mis':
            return 'Missing'
        if key_name == 'extremeness':
            return 'Extremeness'
        if key_name == 'f':
            return 'Multivariate'
        # self.logger('Unexpected dataset name (unknown variable in name)')
        return None

    @staticmethod
    def get_key_and_value(dataset_name):
        # Extract var name and value from dataset name
        var_re = re.compile(r'.+\((\w*)=(.*)\)')
        # e.g. 'Syn Extreme Outliers (pol=0.1)'
        match = var_re.search(dataset_name)
        if not match:
            # self.logger.warn('Unexpected dataset name (not variable in name)')
            return '-', dataset_name
        var_key = match.group(1)
        var_value = match.group(2)
        return Evaluator.translate_var_key(var_key), var_value

    @staticmethod
    def get_dataset_types(mi_df):
        types = mi_df.index.get_level_values('Type')
        indexes = np.unique(types, return_index=True)[1]
        return [types[index] for index in sorted(indexes)]

    @staticmethod
    def insert_multi_index_yaxis(ax, mi_df):
        type_title_offset = -1.6  # depends on string length of xaxis ticklabels

        datasets = mi_df.index
        dataset_types = Evaluator.get_dataset_types(mi_df)  # Returns unique entries keeping original order
        logging.getLogger(__name__).debug('Plotting heatmap for groups {" ".join(dataset_types)}')

        ax.set_yticks(np.arange(len(datasets)))
        ax.set_yticklabels([x[1] for x in datasets])

        y_axis_title_pos = 0  # Store at which position we are for plotting the next title
        for idx, dataset_type in enumerate(dataset_types):
            section_frame = mi_df.iloc[mi_df.index.get_level_values('Type') == dataset_type]
            # Somehow it's sorted by its occurence (which is what we want here)
            dataset_levels = section_frame.index.remove_unused_levels().levels[1]
            title_pos = y_axis_title_pos + 0.5 * (len(dataset_levels) - 1)
            ax.text(type_title_offset, title_pos, dataset_type, ha='center', va='center', rotation=90,
                    fontproperties=FontProperties(weight='bold'))
            if idx < len(dataset_types) - 1:
                sep_pos = y_axis_title_pos + (len(dataset_levels) - 0.6)
                ax.text(-0.5, sep_pos, '_' * int(type_title_offset * -10), ha='right', va='center')
            y_axis_title_pos += len(dataset_levels)

    @staticmethod
    def to_multi_index_frame(evaluators):
        evaluator = evaluators[0]
        for other_evaluator in evaluators[1:]:
            assert evaluator.detectors == other_evaluator.detectors, 'All evaluators should use the same detectors'
        pivot_benchmarks = [ev.benchmark_results.pivot(index='dataset', columns='algorithm',
                                                       values='auroc') for ev in evaluators]

        concat_benchmarks = pd.concat(pivot_benchmarks)
        auroc_matrix = concat_benchmarks.groupby(['dataset']).mean()

        datasets = [[evaluator.get_key_and_value(str(d)) for d in ev.index.values]
                    for ev in pivot_benchmarks]
        datasets = [tuple(d) for d in np.concatenate(datasets)]  # Required for MultiIndex.from_tuples
        datasets = pd.MultiIndex.from_tuples(datasets, names=['Type', 'Level'])
        auroc_matrix.index = datasets
        return auroc_matrix

    def get_multi_index_dataframe(self):
        return Evaluator.to_multi_index_frame([self])

    @staticmethod
    def plot_heatmap(evaluators, store=True):
        mi_df = Evaluator.to_multi_index_frame(evaluators)
        detectors, datasets = mi_df.columns, mi_df.index

        fig, ax = plt.subplots(figsize=(len(detectors) + 2, len(datasets)))
        im = ax.imshow(mi_df, cmap=plt.get_cmap('YlOrRd'), vmin=0, vmax=1)
        plt.colorbar(im)

        # Show MultiIndex for ordinate
        Evaluator.insert_multi_index_yaxis(ax, mi_df)

        # Rotate the tick labels and set their alignment.
        ax.set_xticks(np.arange(len(detectors)))
        ax.set_xticklabels(detectors)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

        # Loop over data dimensions and create text annotations.
        for i in range(len(detectors)):
            for j in range(len(datasets)):
                ax.text(i, j, f'{mi_df.iloc[j, i]:.2f}', ha='center', va='center', color='w',
                        path_effects=[path_effects.withSimplePatchShadow(
                            offset=(1, -1), shadow_rgbFace='b', alpha=0.9)])

        ax.set_title('AUROC over all datasets and detectors')
        # Prevent bug where x axis ticks are completely outside of bounds (matplotlib/issues/5456)
        if len(datasets) > 2:
            fig.tight_layout()
        if store:
            evaluators[0].store(fig, 'heatmap', no_counters=True, store_in_figures=True)
        return fig

    def plot_single_heatmap(self, store=True):
        Evaluator.plot_heatmap([self], store)

    @staticmethod
    def get_printable_runs_results(results):
        print_order = ['dataset', 'algorithm', 'accuracy', 'precision', 'recall', 'F1-score', 'F0.1-score', 'auroc']
        rename_columns = [col for col in print_order if col not in ['dataset', 'algorithm']]

        # calc std and mean for each algorithm per dataset
        std_results = results.groupby(['dataset', 'algorithm']).std(ddof=0).fillna(0)
        # get rid of multi-index
        std_results = std_results.reset_index()
        std_results = std_results[print_order]
        std_results.rename(inplace=True, index=str,
                           columns=dict([(old_col, old_col + '_std') for old_col in rename_columns]))

        avg_results = results.groupby(['dataset', 'algorithm'], as_index=False).mean()
        avg_results = avg_results[print_order]

        avg_results_renamed = avg_results.rename(
            index=str, columns=dict([(old_col, old_col + '_avg') for old_col in rename_columns]))
        return std_results, avg_results, avg_results_renamed

    def gen_merged_tables(self, results, title_suffix=None, store=True):
        title_suffix = f'_{title_suffix}' if title_suffix else ''
        std_results, avg_results, avg_results_renamed = Evaluator.get_printable_runs_results(results)

        ds_title_suffix = f'per_dataset{title_suffix}'
        self.print_merged_table_per_dataset(std_results)
        self.gen_merged_latex_per_dataset(std_results, f'std_{ds_title_suffix}', store=store)

        self.print_merged_table_per_dataset(avg_results_renamed)
        self.gen_merged_latex_per_dataset(avg_results_renamed, f'avg_{ds_title_suffix}', store=store)

        det_title_suffix = f'per_algorithm{title_suffix}'
        self.print_merged_table_per_algorithm(std_results)

        self.gen_merged_latex_per_algorithm(std_results, f'std_{det_title_suffix}', store=store)
        self.print_merged_table_per_algorithm(avg_results_renamed)
        self.gen_merged_latex_per_algorithm(avg_results_renamed, f'avg_{det_title_suffix}', store=store)

    def binarize(self, score, threshold=None):
        threshold = threshold if threshold is not None else self.threshold(score)
        score = np.where(np.isnan(score), np.nanmin(score) - sys.float_info.epsilon, score)
        return np.where(score >= threshold, 1, 0)

    def threshold(self, score):
        return np.nanmean(score) + 2 * np.nanstd(score)


    ##########################################################################

    import logging

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from scipy.stats import multivariate_normal
    from torch.utils.data import DataLoader
    from torch.utils.data.sampler import SubsetRandomSampler
    from tqdm import trange

    from .algorithm_utils import Algorithm, PyTorchUtils

    class LSTMED(Algorithm, PyTorchUtils):
        def __init__(self, name: str = 'LSTM-ED', num_epochs: int = 20, batch_size: int = 16, lr: float = 1e-3,
                     hidden_size: int = 5, sequence_length: int = 30, train_gaussian_percentage: float = 0.20,
                     n_layers: tuple = (1, 1), use_bias: tuple = (True, True), dropout: tuple = (0, 0),
                     seed: int = None, gpu: int = None, details=True, window: int = 15):
            Algorithm.__init__(self, __name__, name, seed, details=details)
            PyTorchUtils.__init__(self, seed, gpu)
            self.num_epochs = num_epochs
            self.batch_size = batch_size
            self.lr = lr
            self.window = window
            self.hidden_size = hidden_size
            self.sequence_length = sequence_length
            self.train_gaussian_percentage = train_gaussian_percentage

            self.n_layers = n_layers
            self.use_bias = use_bias
            self.dropout = dropout

            self.lstmed = None
            self.proto_input_space_ind = None
            self.hidden_and_prototype_as_df = pd.DataFrame(columns=['hidden_and_prototype_sequences', 'indicator'])
            self.mean, self.cov, self.epoch_loss = None, None, None

        def loss_e(self, prototype, enc_hidden):
            k_list = []
            for k in range(prototype.shape[0]):
                b_list = []
                for batch in range(enc_hidden.shape[0]):
                    l = torch.sum(torch.mul(prototype[k] - enc_hidden[batch], prototype[k] - enc_hidden[batch]))
                    b_list.append(l)
                b_list = torch.stack(b_list)
                min_b = torch.min(b_list)
                k_list.append(min_b)
            k_list = torch.stack(k_list)
            return torch.sum(k_list)

        def loss_d(self, prototype, d_min=2.0):
            sum = torch.tensor(0)
            for i in range(prototype.shape[0]):
                for j in range(i + 1, prototype.shape[0]):
                    sum = sum + torch.square(torch.max(torch.tensor(0), d_min - torch.sqrt_(torch.sum(torch.mul(
                        prototype[i] - prototype[j], prototype[i] - prototype[j]), 0))))
            return sum

        def loss_c(self, prototype, enc_hidden):
            b_list = []
            for batch in range(enc_hidden.shape[0]):
                k_list = []
                for k in range(prototype.shape[0]):
                    l = torch.sum(torch.mul(prototype[k] - enc_hidden[batch], prototype[k] - enc_hidden[batch]))
                    k_list.append(l)
                k_list = torch.stack(k_list)
                min_e = torch.min(k_list)
                b_list.append(min_e)
            b_list = torch.stack(b_list)
            return torch.sum(b_list)

        def fit(self, X: pd.DataFrame):
            X.interpolate(inplace=True)
            X.bfill(inplace=True)
            data = X.values
            sequences = [data[i:i + self.sequence_length] for i in
                         range(0, data.shape[0] - self.sequence_length + 1, self.window)]
            indices = np.random.permutation(len(sequences))
            split_point = int(self.train_gaussian_percentage * len(sequences))
            train_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=True,
                                      sampler=SubsetRandomSampler(indices[:-split_point]), pin_memory=True)
            train_gaussian_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=True,
                                               sampler=SubsetRandomSampler(indices[-split_point:]), pin_memory=True)

            self.lstmed = LSTMEDModule(X.shape[1], self.hidden_size,
                                       self.n_layers, self.use_bias, self.dropout,
                                       seed=self.seed, gpu=self.gpu)
            self.to_device(self.lstmed)
            optimizer = torch.optim.Adam(self.lstmed.parameters(), lr=self.lr)
            self.lstmed.train()
            epoch_loss = []
            for epoch in trange(self.num_epochs):
                logging.debug(f'Epoch {epoch + 1}/{self.num_epochs}.')
                loss_arr = []
                for ts_batch in train_loader:
                    output, enc_hidden, a = self.lstmed(self.to_var(ts_batch), return_latent=True)
                    loss_c = self.loss_c(self.lstmed.prototype_layer.prototype, enc_hidden)
                    loss_d = self.loss_d(self.lstmed.prototype_layer.prototype, d_min=2.0)
                    loss_e = self.loss_e(self.lstmed.prototype_layer.prototype, enc_hidden)
                    # loss_w = torch.sum(torch.abs(self.lstmed.hidden2output.weight))
                    loss = nn.MSELoss(size_average=False)(output, self.to_var(ts_batch.float()))
                    total_loss = loss + loss_c + loss_d + loss_e
                    loss_arr.append(total_loss)
                    self.lstmed.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                epoch_loss.append(sum(loss_arr))

            a_min = torch.ones(self.lstmed.prototype_layer.k)
            self.proto_input_space_ind = torch.Tensor(self.lstmed.prototype_layer.k, 2)

            for i in range(len(sequences)):
                output, enc_hidden, a = self.lstmed(
                    self.to_var(torch.Tensor(sequences[i]).expand(1, sequences[i].shape[0], sequences[i].shape[1])),
                    return_latent=True)
                self.hidden_and_prototype_as_df.loc[len(self.hidden_and_prototype_as_df)] = [
                    enc_hidden[0].detach().numpy().tolist(), 0]

                for k in range(a.shape[1]):
                    if a_min[k] > a[torch.argmin(a[:, k]), k]:
                        a_min[k] = a[torch.argmin(a[:, k]), k]
                        self.proto_input_space_ind[k, :] = torch.Tensor([i, i + self.sequence_length])
            for k in range(self.lstmed.prototype_layer.prototype.shape[0]):
                self.hidden_and_prototype_as_df.loc[len(self.hidden_and_prototype_as_df)] = [
                    self.lstmed.prototype_layer.prototype[k].detach().numpy().tolist(), 1]
            # print(self.proto_input_space_ind)
            # print(self.hidden_and_prototype_as_df)

            self.lstmed.eval()
            error_vectors = []
            for ts_batch in train_gaussian_loader:
                output = self.lstmed(self.to_var(ts_batch))
                error = nn.L1Loss(reduce=False)(output, self.to_var(ts_batch.float()))
                error_vectors += list(error.view(-1, X.shape[1]).data.cpu().numpy())

            self.mean = np.mean(error_vectors, axis=0)
            self.cov = np.cov(error_vectors, rowvar=False)
            self.epoch_loss = epoch_loss

        def predict(self, X: pd.DataFrame):
            X.interpolate(inplace=True)
            X.bfill(inplace=True)
            data = X.values
            sequences = [data[i:i + self.sequence_length] for i in
                         range(0, data.shape[0] - self.sequence_length + 1, self.window)]
            data_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, shuffle=False, drop_last=False)
            print(7)
            print(data.shape[0])
            print(len(data_loader))
            print(len(sequences))

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
                lattice[i % self.sequence_length, i:i + self.sequence_length] = score
            scores = np.nanmean(lattice, axis=0)
            print(scores)

            if self.details:
                outputs = np.concatenate(outputs)
                lattice = np.full((self.sequence_length, X.shape[0], X.shape[1]), np.nan)
                for i, output in enumerate(outputs):
                    lattice[i % self.sequence_length, i:i + self.sequence_length, :] = output
                self.prediction_details.update({'reconstructions_mean': np.nanmean(lattice, axis=0).T})

                errors = np.concatenate(errors)
                lattice = np.full((self.sequence_length, X.shape[0], X.shape[1]), np.nan)
                for i, error in enumerate(errors):
                    lattice[i % self.sequence_length, i:i + self.sequence_length, :] = error
                self.prediction_details.update({'errors_mean': np.nanmean(lattice, axis=0).T})

            return scores

    class prototype_layer(nn.Module, PyTorchUtils):
        def __init__(self, hidden_size: int, seed: int, gpu: int, k=2):
            super().__init__()
            PyTorchUtils.__init__(self, seed, gpu)
            self.hidden_size = hidden_size
            self.k = k
            self.prototype_size = torch.Tensor(k, hidden_size)
            self.init_values = nn.init.uniform(self.prototype_size, a=0.0, b=1.0)
            self.prototype = nn.Parameter(self.init_values)
            self.similarity2output = nn.Linear(self.k, 2)

        def forward(self, x, batch_size):
            a = torch.zeros((batch_size, self.k))
            d = x[0].unsqueeze(1) - self.prototype.unsqueeze(1)
            for i in range(self.k):
                a[:, i] = torch.exp(-torch.sum(torch.mul(d[0, i], d[0, i]), 1))
            return a

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

            self.encoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
                                   num_layers=self.n_layers[0], bias=self.use_bias[0], dropout=self.dropout[0])
            self.to_device(self.encoder)

            self.decoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
                                   num_layers=self.n_layers[1], bias=self.use_bias[1], dropout=self.dropout[1])
            self.to_device(self.decoder)
            # -----------
            self.prototype_layer = prototype_layer(self.hidden_size, seed, gpu, k=8)
            self.to_device(self.prototype_layer)
            # -----------
            self.hidden2output = nn.Linear(self.hidden_size, self.n_features)
            self.to_device(self.hidden2output)

        def _init_hidden(self, batch_size):
            return (self.to_var(torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_()),
                    self.to_var(torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_()))

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
                output[:, i, :] = self.hidden2output(dec_hidden[0][0, :])

                if self.training:
                    _, dec_hidden = self.decoder(ts_batch[:, i].unsqueeze(1).float(), dec_hidden)
                else:
                    _, dec_hidden = self.decoder(output[:, i].unsqueeze(1), dec_hidden)

            a = self.prototype_layer(enc_hidden, batch_size)

            return (output, enc_hidden[1][-1], a) if return_latent else output

##########################################################################################
import glob
import os

import numpy as np
import pandas as pd

from src.algorithms import GRUED
from src.datasets import RealPickledDataset
from src.evaluation import Evaluator



RUNS = 1


def main():
    evaluate_real_datasets()


def detectors(seed,step,sequence_length):
    standard_epochs = 1
    dets = [
            GRUED(num_epochs=standard_epochs, seed=seed, step=step, sequence_length=sequence_length)
            ]

    return sorted(dets, key=lambda x: x.framework)


def evaluate_real_datasets():
    REAL_DATASET_GROUP_PATH = 'data/raw/'
    real_dataset_groups = glob.glob(REAL_DATASET_GROUP_PATH + '*')
    seeds = [3424441233]
    results = pd.DataFrame()
    datasets = []
    for real_dataset_group in real_dataset_groups:
        for data_set_path in glob.glob(real_dataset_group + '/labeled/train/*'):
            data_set_name = data_set_path.split('/')[-1].replace('.pkl', '')
            dataset = RealPickledDataset(data_set_name, data_set_path)
            datasets.append(dataset)

    for seed in seeds:
        evaluator = Evaluator(datasets, detectors, seed=seed, step=15, sequence_length=30)
        evaluator.evaluate()
        result = evaluator.benchmarks()
        evaluator.plot_roc_curves()
        evaluator.plot_threshold_comparison()
        evaluator.plot_scores()
        results = results.append(result, ignore_index=True)

    avg_results = results.groupby(['dataset', 'algorithm'], as_index=False).mean()
    evaluator.set_benchmark_results(avg_results)
    evaluator.export_results('run_real_datasets')
    evaluator.create_boxplots(runs=RUNS, data=results, detectorwise=False)
    evaluator.create_boxplots(runs=RUNS, data=results, detectorwise=True)



if __name__ == '__main__':
    main()


''''##########'''
import gc
import logging
import os
import pickle
import re
import sys
import traceback
from textwrap import wrap
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
import progressbar
import time
from sklearn.metrics import accuracy_score, fbeta_score
from sklearn.metrics import precision_recall_fscore_support as prf
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.manifold._t_sne import TSNE
from tabulate import tabulate

from .config import init_logging


class Evaluator:
    def __init__(self, datasets: list, detectors: callable,step, sequence_length, output_dir: {str} = None, seed: int = None,
                 create_log_file=True):
        """
        :param datasets: list of datasets
        :param detectors: callable that returns list of detectors
        """
        assert np.unique([x.name for x in datasets]).size == len(datasets), 'Some datasets have the same name!'
        self.datasets = datasets
        self._detectors = detectors
        self.step = step
        self.sequence_length = sequence_length
        self.output_dir = output_dir or 'reports'
        self.results = dict()
        self.proto_input_space_ind = dict()
        if create_log_file:
            init_logging(os.path.join(self.output_dir, 'logs'))
        self.logger = logging.getLogger(__name__)
        # Dirty hack: Is set by the main.py to insert results from multiple evaluator runs
        self.benchmark_results = None
        # Last passed seed value in evaluate()
        self.seed = seed

    @property
    def detectors(self):
        detectors = self._detectors(self.seed,self.step,self.sequence_length)
        assert np.unique([x.name for x in detectors]).size == len(detectors), 'Some detectors have the same name!'
        return detectors

    def set_benchmark_results(self, benchmark_result):
        self.benchmark_results = benchmark_result

    def export_results(self, name):
        output_dir = os.path.join(self.output_dir, 'evaluators')
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime('%Y-%m-%d-%H%M%S')
        path = os.path.join(output_dir, f'{name}-{timestamp}.pkl')
        path1 = os.path.join(output_dir, f'{name}-{timestamp}-benchmark_results.csv')
        self.logger.info(f'Store evaluator results at {os.path.abspath(path)}')
        save_dict = {
            'datasets': [x.name for x in self.datasets],
            'detectors': [x.name for x in self.detectors],
            'benchmark_results': self.benchmark_results,
            'results': self.results,
            'output_dir': self.output_dir,
            'seed': int(self.seed),
        }
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
            print(save_dict['benchmark_results'].to_string())
        return path


    # Import benchmark_results if this evaluator uses the same detectors and datasets
    # self.results are not available because they are overwritten by each run
    def import_results(self, name):
        output_dir = os.path.join(self.output_dir, 'evaluators')
        path = os.path.join(output_dir, f'{name}.pkl')
        self.logger.info(f'Read evaluator results at {os.path.abspath(path)}')
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)

        self.logger.debug(f'Importing detectors {"; ".join(save_dict["detectors"])}')
        my_detectors = [x.name for x in self.detectors]
        assert np.array_equal(save_dict['detectors'], my_detectors), 'Detectors should be the same'

        self.logger.debug(f'Importing datasets {"; ".join(save_dict["datasets"])}')
        my_datasets = [x.name for x in self.datasets]
        assert np.array_equal(save_dict['datasets'], my_datasets), 'Datasets should be the same'

        self.benchmark_results = save_dict['benchmark_results']
        self.seed = save_dict['seed']
        self.results = save_dict['results']

    @staticmethod
    def get_accuracy_precision_recall_fscore(y_true: list, y_pred: list):
        accuracy = accuracy_score(y_true, y_pred)
        # warn_for=() avoids log warnings for any result being zero
        precision, recall, f_score, _ = prf(y_true, y_pred, average='binary', warn_for=())
        if precision == 0 and recall == 0:
            f01_score = 0
        else:
            f01_score = fbeta_score(y_true, y_pred, average='binary', beta=0.1)
        return accuracy, precision, recall, f_score, f01_score

    @staticmethod
    def get_auroc(det, ds, score):
        if np.isnan(score).all():
            score = np.zeros_like(score)
        _, _, _, y_test = ds.data()
        score_nonan = score.copy()
        # Rank NaN below every other value in terms of anomaly score
        score_nonan[np.isnan(score_nonan)] = np.nanmin(score_nonan) - sys.float_info.epsilon
        fpr, tpr, _ = roc_curve(y_test, score_nonan)
        return auc(fpr, tpr)

    def get_optimal_threshold(self, det, y_test, score, steps=100, return_metrics=False):
        maximum = np.nanmax(score)
        minimum = np.nanmin(score)
        threshold = np.linspace(minimum, maximum, steps)
        metrics = list(self.get_metrics_by_thresholds(y_test, score, threshold))
        metrics = np.array(metrics).T
        anomalies, acc, prec, rec, f_score, f01_score = metrics
        if return_metrics:
            return anomalies, acc, prec, rec, f_score, f01_score, threshold
        else:
            return threshold[np.argmax(f01_score)]

    def evaluate(self):
        for ds in progressbar.progressbar(self.datasets):
            (X_train, y_train, X_test, y_test) = ds.data()
            for det in progressbar.progressbar(self.detectors):
                self.logger.info(f'Training {det.name} on {ds.name} with seed {self.seed}')
                try:
                    det.fit(X_train.copy())
                    score = det.predict(X_test.copy())
                    self.results[(ds.name, det.name)] = score
                    self.proto_input_space_ind[(ds.name, det.name)] = det.proto_input_space_ind
                    self.plot_prototypes_in_input_space(det, ds)
                    try:
                        self.plot_latents_and_prototypes(det,ds,det.hidden_and_prototype_as_df)
                        self.plot_details(det, ds, score)
                        self.plot_epoch_loss(det,ds)
                    except Exception:
                        pass
                except Exception as e:
                    self.logger.error(f'An exception occurred while training {det.name} on {ds}: {e}')
                    self.logger.error(traceback.format_exc())
                    self.results[(ds.name, det.name)] = np.zeros_like(y_test)
            gc.collect()

    def benchmarks(self) -> pd.DataFrame:
        df = pd.DataFrame()
        for ds in self.datasets:
            _, _, _, y_test = ds.data()
            for det in self.detectors:
                score = self.results[(ds.name, det.name)]
                y_pred = self.binarize(score, self.get_optimal_threshold(det, y_test, np.array(score)))
                acc, prec, rec, f1_score, f01_score = self.get_accuracy_precision_recall_fscore(y_test, y_pred)
                confusion_mat = confusion_matrix(y_test,y_pred,labels=[0,1])
                self.plot_confusion_matrix(det,ds,confusion_mat)
                score = self.results[(ds.name, det.name)]
                auroc = self.get_auroc(det, ds, score)
                df = df.append({'dataset': ds.name,
                                'algorithm': det.name,
                                'accuracy': acc,
                                'precision': prec,
                                'recall': rec,
                                'F1-score': f1_score,
                                'F0.1-score': f01_score,
                                'auroc': auroc,
                                'confusion_mat':confusion_mat
                                },
                               ignore_index=True)
        return df

    # plot_prototypes_in_input_space plots/ highlights the input sequences which are closer to the prototypes with respect
    # the distance measure 'a= exp(-d)' in the hidden space.
    def plot_prototypes_in_input_space(self, det, ds, store = True):
        plt.close('all')
        fig = plt.figure()
        X_train, y_train, X_test, y_test = ds.data()
        n_proto = self.proto_input_space_ind[(ds.name, det.name)].shape[0]
        for k in range(n_proto):
            st_ind = int(self.proto_input_space_ind[(ds.name, det.name)][k, 0].item()) * self.step
            seq_len = self.sequence_length
            end_ind = st_ind + seq_len
            for col in X_train.columns:
                plt.subplot(n_proto,1,k+1)
                plt.plot(X_train[col].iloc[(st_ind-(seq_len*1)):(end_ind+(seq_len*1))])
                plt.plot(X_train[col].iloc[st_ind:end_ind])
                plt.ylabel(f'Proto-{k}')
        if store:
            self.store(fig, f'inputs_closest_to_prototypes-{det.name}-{ds.name}')
        return fig

    # plot_latents_and_prototypes function gathers the hidden space for the inputs along with the prototypes and maps them
    # to 2 dimensions by using tnse approach of dimensionality reduction and plots them as a points in the latent space.
    def plot_latents_and_prototypes(self, det, ds, df, store = True):
        plt.close('all')
        df1 = df.hidden_and_prototype_sequences.apply(pd.Series).assign(**{'indicator':df.indicator})
        tnse = TSNE(n_components=2, verbose=0, perplexity=5, n_iter=300)
        tnse_results = tnse.fit_transform(df1.iloc[:, :-1])
        fig = plt.figure()
        plt.scatter(tnse_results[:, 0], tnse_results[:, 1], marker='.', c=df1.indicator, cmap='bwr_r')
        if store:
            self.store(fig, f'latents_plot-{det.name}-{ds.name}')
        return fig

    def plot_confusion_matrix(self,det,ds,conf_matrix,store=True):
        plt.close('all')
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

        plt.xlabel('Predictions', fontsize=18)
        plt.ylabel('Actuals', fontsize=18)
        plt.title('Confusion Matrix', fontsize=18)
        if store:
            self.store(fig, f'confusion_matrix-{det.name}-{ds.name}')
        return fig

    def plot_epoch_loss(self,det,ds,store=True):
        plt.close('all')
        fig = plt.figure()
        arr = []
        for i in det.epoch_loss:
            arr.append(i.item())

        plt.plot(arr)
        if store:
            self.store(fig, f'epoch_loss_plot-{det.name}-{ds.name}')
        return fig

    def get_metrics_by_thresholds(self, y_test: list, score: list, thresholds: list):
        for threshold in thresholds:
            anomaly = self.binarize(score, threshold=threshold)
            metrics = Evaluator.get_accuracy_precision_recall_fscore(y_test, anomaly)
            yield (anomaly.sum(), *metrics)

    def plot_scores(self, store=True):
        detectors = self.detectors
        plt.close('all')
        figures = []
        for ds in self.datasets:
            X_train, y_train, X_test, y_test = ds.data()
            subtitle_loc = 'left'
            fig = plt.figure(figsize=(15, 15))
            fig.canvas.set_window_title(ds.name)

            sp = fig.add_subplot((2 * len(detectors) + 3), 1, 1)
            sp.set_title('original training data', loc=subtitle_loc)
            for col in X_train.columns:
                plt.plot(X_train[col], alpha=0.5)
            for det in detectors:
                for k in range(self.proto_input_space_ind[(ds.name, det.name)].shape[0]):
                        st_ind = int(self.proto_input_space_ind[(ds.name, det.name)][k, 0].item()) * self.step
                        seq_len = self.sequence_length
                        end_ind = st_ind + seq_len
                        for col in X_train.columns:
                            plt.plot(X_train[col].iloc[st_ind:end_ind])
                            plt.axvspan(st_ind, end_ind, color='black')
            sp = fig.add_subplot((2 * len(detectors) + 3), 1, 2)
            sp.set_title('original test set', loc=subtitle_loc)
            for col in X_test.columns:
                plt.plot(X_test[col])

            sp = fig.add_subplot((2 * len(detectors) + 3), 1, 3)
            sp.set_title('binary labels of test set', loc=subtitle_loc)
            plt.plot(y_test)

            subplot_num = 4
            for det in detectors:
                sp = fig.add_subplot((2 * len(detectors) + 3), 1, subplot_num)
                sp.set_title(f'scores of {det.name}', loc=subtitle_loc)
                score = self.results[(ds.name, det.name)]
                plt.plot(np.arange(len(score)), [x for x in score])
                threshold_line = len(score) * [self.get_optimal_threshold(det, y_test, np.array(score))]
                plt.plot([x for x in threshold_line])
                subplot_num += 1

                sp = fig.add_subplot((2 * len(detectors) + 3), 1, subplot_num)
                sp.set_title(f'binary labels of {det.name}', loc=subtitle_loc)
                plt.plot(np.arange(len(score)),
                         [x for x in self.binarize(score, self.get_optimal_threshold(det, y_test, np.array(score)))])
                subplot_num += 1
            fig.subplots_adjust(top=0.9, hspace=0.4)
            fig.tight_layout()
            if store:
                self.store(fig, f'scores_{ds.name}')
            figures.append(fig)
        return figures

    def plot_threshold_comparison(self, steps=40, store=True):
        detectors = self.detectors
        plt.close('all')
        plots_shape = len(detectors), len(self.datasets)
        fig, axes = plt.subplots(*plots_shape, figsize=(len(detectors) * 15, len(self.datasets) * 5))
        # Ensure two dimensions for iteration
        axes = np.array(axes).reshape(*plots_shape).T
        plt.suptitle('Compare thresholds', fontsize=10)
        for ds, axes_row in zip(self.datasets, axes):
            _, _, X_test, y_test = ds.data()

            for det, ax in zip(detectors, axes_row):
                score = np.array(self.results[(ds.name, det.name)])

                anomalies, _, prec, rec, f_score, f01_score, thresh = self.get_optimal_threshold(
                    det, y_test, score, return_metrics=True)

                ax.plot(thresh, anomalies / len(y_test),
                        label=fr'anomalies ({len(y_test)} $\rightarrow$ 1)')
                ax.plot(thresh, prec, label='precision')
                ax.plot(thresh, rec, label='recall')
                ax.plot(thresh, f_score, label='f_score', linestyle='dashed')
                ax.plot(thresh, f01_score, label='f01_score', linestyle='dashed')
                ax.set_title(f'{det.name} on {ds.name}')
                ax.set_xlabel('Threshold')
                ax.legend()

        # Avoid overlapping title and axis labels
        plt.xlim([0.0, 1.0])
        fig.subplots_adjust(top=0.9, hspace=0.4, right=1, left=0)
        fig.tight_layout()
        if store:
            self.store(fig, 'metrics_by_thresholds')
        return fig

    def plot_roc_curves(self, store=True):
        detectors = self.detectors
        plt.close('all')
        figures = []
        for ds in self.datasets:
            _, _, _, y_test = ds.data()
            fig_scale = 3
            fig = plt.figure(figsize=(fig_scale * len(detectors), fig_scale))
            fig.canvas.set_window_title(ds.name + ' ROC')
            fig.suptitle(f'ROC curve on {ds.name}', fontsize=14, y='1.1')
            subplot_count = 1
            for det in detectors:
                self.logger.info(f'Plotting ROC curve for {det.name} on {ds.name}')
                score = self.results[(ds.name, det.name)]
                if np.isnan(score).all():
                    score = np.zeros_like(score)
                # Rank NaN below every other value in terms of anomaly score
                score[np.isnan(score)] = np.nanmin(score) - sys.float_info.epsilon
                fpr, tpr, _ = roc_curve(y_test, score)
                roc_auc = auc(fpr, tpr)
                plt.subplot(1, len(detectors), subplot_count)
                plt.plot(fpr, tpr, color='darkorange',
                         lw=2, label='area = %0.2f' % roc_auc)
                subplot_count += 1
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.gca().set_aspect('equal', adjustable='box')
                plt.title('\n'.join(wrap(det.name, 20)))
                plt.legend(loc='lower right')
            plt.tight_layout()
            if store:
                self.store(fig, f'roc_{ds.name}')
            figures.append(fig)
        return figures

    def plot_auroc(self, store=True, title='AUROC'):
        plt.close('all')
        self.benchmark_results[['dataset', 'algorithm', 'auroc']].pivot(
            index='algorithm', columns='dataset', values='auroc').plot(kind='bar')
        plt.legend(loc=3, framealpha=0.5)
        plt.xticks(rotation=20)
        plt.ylabel('AUC', rotation='horizontal', labelpad=20)
        plt.title(title)
        plt.ylim(ymin=0, ymax=1)
        plt.tight_layout()
        if store:
            self.store(plt.gcf(), 'auroc', store_in_figures=True)

    def plot_details(self, det, ds, score, store=True):
        if not det.details:
            return
        plt.close('all')
        cmap = plt.get_cmap('inferno')
        _, _, X_test, y_test = ds.data()

        grid = 0
        for value in det.prediction_details.values():
            grid += 1 if value.ndim == 1 else value.shape[0]
        grid += X_test.shape[1]  # data
        grid += 1 + 1  # score and gt

        fig, axes = plt.subplots(grid, 1, figsize=(15, 1.5 * grid))

        i = 0
        c = cmap(i / grid)
        axes[i].set_title('test data')
        for col in X_test.values.T:
            axes[i].plot(col, color=c)
            i += 1
        c = cmap(i / grid)

        axes[i].set_title('test gt data')
        axes[i].plot(y_test.values, color=c)
        i += 1
        c = cmap(i / grid)

        axes[i].set_title('scores')
        axes[i].plot(score, color=c)
        i += 1
        c = cmap(i / grid)

        for key, values in det.prediction_details.items():
            axes[i].set_title(key)
            if values.ndim == 1:
                axes[i].plot(values, color=c)
                i += 1
            elif values.ndim == 2:
                for v in values:
                    axes[i].plot(v, color=c)
                    i += 1
            else:
                self.logger.warning('plot_details: not sure what to do')
            c = cmap(i / grid)

        fig.tight_layout()
        if store:
            self.store(fig, f'details_{det.name}_{ds.name}')
        return fig

    # create boxplot diagrams for auc values for each algorithm/dataset per algorithm/dataset
    def create_boxplots(self, runs, data, detectorwise=True, store=True):
        target = 'algorithm' if detectorwise else 'dataset'
        grouped_by = 'dataset' if detectorwise else 'algorithm'
        relevant_results = data[['algorithm', 'dataset', 'auroc']]
        figures = []
        for det_or_ds in (self.detectors if detectorwise else self.datasets):
            relevant_results[relevant_results[target] == det_or_ds.name].boxplot(by=grouped_by, figsize=(15, 15))
            plt.suptitle('')  # boxplot() adds a suptitle
            plt.title(f'AUC grouped by {grouped_by} for {det_or_ds.name} over {runs} runs')
            plt.ylim(ymin=0, ymax=1)
            plt.tight_layout()
            figures.append(plt.gcf())
            if store:
                self.store(plt.gcf(), f'boxplot_auc_for_{det_or_ds.name}_{runs}_runs', store_in_figures=True)
        return figures

    # create bar charts for averaged pipeline results per algorithm/dataset
    def create_bar_charts(self, runs, detectorwise=True, store=True):
        target = 'algorithm' if detectorwise else 'dataset'
        grouped_by = 'dataset' if detectorwise else 'algorithm'
        relevant_results = self.benchmark_results[['algorithm', 'dataset', 'auroc']]
        figures = []
        for det_or_ds in (self.detectors if detectorwise else self.datasets):
            relevant_results[relevant_results[target] == det_or_ds.name].plot(x=grouped_by, kind='bar', figsize=(7, 7))
            plt.suptitle('')  # boxplot() adds a suptitle
            plt.title(f'AUC for {target} {det_or_ds.name} over {runs} runs')
            plt.ylim(ymin=0, ymax=1)
            plt.tight_layout()
            figures.append(plt.gcf())
            if store:
                self.store(plt.gcf(), f'barchart_auc_for_{det_or_ds.name}_{runs}_runs', store_in_figures=True)
        return figures

    def store(self, fig, title, extension='pdf', no_counters=False, store_in_figures=False):
        timestamp = time.strftime('%Y-%m-%d-%H%M%S')
        if store_in_figures:
            output_dir = os.path.join(self.output_dir, 'figures',title)
        else:
            output_dir = os.path.join(self.output_dir, 'figures', f'seed-{self.seed}',title)
        os.makedirs(output_dir, exist_ok=True)
        counters_str = '' if no_counters else f'-{len(self.detectors)}-{len(self.datasets)}'
        path = os.path.join(output_dir, f'{counters_str}--{timestamp}.{extension}')
        fig.savefig(path)
        self.logger.info(f'Stored plot at {path}')

    def store_text(self, content, title, extension='txt'):
        timestamp = int(time.time())
        output_dir = os.path.join(self.output_dir, 'tables', f'seed-{self.seed}')
        path = os.path.join(output_dir, f'{title}-{len(self.detectors)}-{len(self.datasets)}-{timestamp}.{extension}')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(content)
        self.logger.info(f'Stored {extension} file at {path}')

    def print_merged_table_per_dataset(self, results):
        for ds in self.datasets:
            table = tabulate(results[results['dataset'] == ds.name], headers='keys', tablefmt='psql')
            self.logger.info(f'Dataset: {ds.name}\n{table}')

    def gen_merged_latex_per_dataset(self, results, title_suffix=None, store=True):
        title = f'latex_merged{f"_{title_suffix}" if title_suffix else ""}'
        content = ''
        for ds in self.datasets:
            content += f'''{ds.name}:\n\n{tabulate(results[results['dataset'] == ds.name],
                                                   headers='keys', tablefmt='latex')}\n\n'''
        if store:
            self.store_text(content=content, title=title, extension='tex')
        return content

    def print_merged_table_per_algorithm(self, results):
        for det in self.detectors:
            table = tabulate(results[results['algorithm'] == det.name], headers='keys', tablefmt='psql')
            self.logger.info(f'Detector: {det.name}\n{table}')

    def gen_merged_latex_per_algorithm(self, results, title_suffix=None, store=True):
        title = f'latex_merged{f"_{title_suffix}" if title_suffix else ""}'
        content = ''
        for det in self.detectors:
            content += f'''{det.name}:\n\n{tabulate(results[results['algorithm'] == det.name],
                                   headers='keys', tablefmt='latex')}\n\n'''
        if store:
            self.store_text(content=content, title=title, extension='tex')
        return content

    @staticmethod
    def translate_var_key(key_name):
        if key_name == 'pol':
            return 'Pollution'
        if key_name == 'mis':
            return 'Missing'
        if key_name == 'extremeness':
            return 'Extremeness'
        if key_name == 'f':
            return 'Multivariate'
        # self.logger('Unexpected dataset name (unknown variable in name)')
        return None

    @staticmethod
    def get_key_and_value(dataset_name):
        # Extract var name and value from dataset name
        var_re = re.compile(r'.+\((\w*)=(.*)\)')
        # e.g. 'Syn Extreme Outliers (pol=0.1)'
        match = var_re.search(dataset_name)
        if not match:
            # self.logger.warn('Unexpected dataset name (not variable in name)')
            return '-', dataset_name
        var_key = match.group(1)
        var_value = match.group(2)
        return Evaluator.translate_var_key(var_key), var_value

    @staticmethod
    def get_dataset_types(mi_df):
        types = mi_df.index.get_level_values('Type')
        indexes = np.unique(types, return_index=True)[1]
        return [types[index] for index in sorted(indexes)]

    @staticmethod
    def insert_multi_index_yaxis(ax, mi_df):
        type_title_offset = -1.6  # depends on string length of xaxis ticklabels

        datasets = mi_df.index
        dataset_types = Evaluator.get_dataset_types(mi_df)  # Returns unique entries keeping original order
        logging.getLogger(__name__).debug('Plotting heatmap for groups {" ".join(dataset_types)}')

        ax.set_yticks(np.arange(len(datasets)))
        ax.set_yticklabels([x[1] for x in datasets])

        y_axis_title_pos = 0  # Store at which position we are for plotting the next title
        for idx, dataset_type in enumerate(dataset_types):
            section_frame = mi_df.iloc[mi_df.index.get_level_values('Type') == dataset_type]
            # Somehow it's sorted by its occurence (which is what we want here)
            dataset_levels = section_frame.index.remove_unused_levels().levels[1]
            title_pos = y_axis_title_pos + 0.5 * (len(dataset_levels) - 1)
            ax.text(type_title_offset, title_pos, dataset_type, ha='center', va='center', rotation=90,
                    fontproperties=FontProperties(weight='bold'))
            if idx < len(dataset_types) - 1:
                sep_pos = y_axis_title_pos + (len(dataset_levels) - 0.6)
                ax.text(-0.5, sep_pos, '_' * int(type_title_offset * -10), ha='right', va='center')
            y_axis_title_pos += len(dataset_levels)

    @staticmethod
    def to_multi_index_frame(evaluators):
        evaluator = evaluators[0]
        for other_evaluator in evaluators[1:]:
            assert evaluator.detectors == other_evaluator.detectors, 'All evaluators should use the same detectors'
        pivot_benchmarks = [ev.benchmark_results.pivot(index='dataset', columns='algorithm',
                                                       values='auroc') for ev in evaluators]

        concat_benchmarks = pd.concat(pivot_benchmarks)
        auroc_matrix = concat_benchmarks.groupby(['dataset']).mean()

        datasets = [[evaluator.get_key_and_value(str(d)) for d in ev.index.values]
                    for ev in pivot_benchmarks]
        datasets = [tuple(d) for d in np.concatenate(datasets)]  # Required for MultiIndex.from_tuples
        datasets = pd.MultiIndex.from_tuples(datasets, names=['Type', 'Level'])
        auroc_matrix.index = datasets
        return auroc_matrix

    def get_multi_index_dataframe(self):
        return Evaluator.to_multi_index_frame([self])

    @staticmethod
    def plot_heatmap(evaluators, store=True):
        mi_df = Evaluator.to_multi_index_frame(evaluators)
        detectors, datasets = mi_df.columns, mi_df.index

        fig, ax = plt.subplots(figsize=(len(detectors) + 2, len(datasets)))
        im = ax.imshow(mi_df, cmap=plt.get_cmap('YlOrRd'), vmin=0, vmax=1)
        plt.colorbar(im)

        # Show MultiIndex for ordinate
        Evaluator.insert_multi_index_yaxis(ax, mi_df)

        # Rotate the tick labels and set their alignment.
        ax.set_xticks(np.arange(len(detectors)))
        ax.set_xticklabels(detectors)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

        # Loop over data dimensions and create text annotations.
        for i in range(len(detectors)):
            for j in range(len(datasets)):
                ax.text(i, j, f'{mi_df.iloc[j, i]:.2f}', ha='center', va='center', color='w',
                        path_effects=[path_effects.withSimplePatchShadow(
                            offset=(1, -1), shadow_rgbFace='b', alpha=0.9)])

        ax.set_title('AUROC over all datasets and detectors')
        # Prevent bug where x axis ticks are completely outside of bounds (matplotlib/issues/5456)
        if len(datasets) > 2:
            fig.tight_layout()
        if store:
            evaluators[0].store(fig, 'heatmap', no_counters=True, store_in_figures=True)
        return fig

    def plot_single_heatmap(self, store=True):
        Evaluator.plot_heatmap([self], store)

    @staticmethod
    def get_printable_runs_results(results):
        print_order = ['dataset', 'algorithm', 'accuracy', 'precision', 'recall', 'F1-score', 'F0.1-score', 'auroc']
        rename_columns = [col for col in print_order if col not in ['dataset', 'algorithm']]

        # calc std and mean for each algorithm per dataset
        std_results = results.groupby(['dataset', 'algorithm']).std(ddof=0).fillna(0)
        # get rid of multi-index
        std_results = std_results.reset_index()
        std_results = std_results[print_order]
        std_results.rename(inplace=True, index=str,
                           columns=dict([(old_col, old_col + '_std') for old_col in rename_columns]))

        avg_results = results.groupby(['dataset', 'algorithm'], as_index=False).mean()
        avg_results = avg_results[print_order]
        avg_results_renamed = avg_results.rename(
            index=str, columns=dict([(old_col, old_col + '_avg') for old_col in rename_columns]))
        return std_results, avg_results, avg_results_renamed

    def gen_merged_tables(self, results, title_suffix=None, store=True):
        title_suffix = f'_{title_suffix}' if title_suffix else ''
        std_results, avg_results, avg_results_renamed = Evaluator.get_printable_runs_results(results)

        ds_title_suffix = f'per_dataset{title_suffix}'
        self.print_merged_table_per_dataset(std_results)
        self.gen_merged_latex_per_dataset(std_results, f'std_{ds_title_suffix}', store=store)

        self.print_merged_table_per_dataset(avg_results_renamed)
        self.gen_merged_latex_per_dataset(avg_results_renamed, f'avg_{ds_title_suffix}', store=store)

        det_title_suffix = f'per_algorithm{title_suffix}'
        self.print_merged_table_per_algorithm(std_results)

        self.gen_merged_latex_per_algorithm(std_results, f'std_{det_title_suffix}', store=store)
        self.print_merged_table_per_algorithm(avg_results_renamed)
        self.gen_merged_latex_per_algorithm(avg_results_renamed, f'avg_{det_title_suffix}', store=store)

    def binarize(self, score, threshold=None):
        threshold = threshold if threshold is not None else self.threshold(score)
        score = np.where(np.isnan(score), np.nanmin(score) - sys.float_info.epsilon, score)
        return np.where(score >= threshold, 1, 0)

    def threshold(self, score):
        return np.nanmean(score) + 2 * np.nanstd(score)
###########################################################################

import glob
import os

import numpy as np
import pandas as pd

from src.algorithms import GRUED
from src.datasets import RealPickledDataset
from src.evaluation import Evaluator
#from tslearn.datasets import UCR_UEA_datasets


RUNS = 1


def main():
    evaluate_real_datasets()


def detectors(seed,step,sequence_length):
    standard_epochs = 30
    dets = [
            GRUED(num_epochs=standard_epochs, seed=seed, step=step, sequence_length=sequence_length)
            ]
    return sorted(dets, key=lambda x: x.framework)


def evaluate_real_datasets():
    REAL_DATASET_GROUP_PATH = 'data/raw/'
    real_dataset_groups = glob.glob(REAL_DATASET_GROUP_PATH + '*')
    seeds = [3424441240]
    results = pd.DataFrame()
    datasets = []
    for real_dataset_group in real_dataset_groups:
        for data_set_path in glob.glob(real_dataset_group + '/labeled/train/*'):
            data_set_name = data_set_path.split('/')[-1].replace('.pkl', '')
            dataset = RealPickledDataset(data_set_name, data_set_path)
            datasets.append(dataset)

    for seed in seeds:
        evaluator = Evaluator(datasets, detectors, seed=seed, step=30, sequence_length=30)
        evaluator.evaluate()
        result = evaluator.benchmarks()
        evaluator.plot_roc_curves()
        evaluator.plot_threshold_comparison()
        evaluator.plot_scores()
        results = results.append(result, ignore_index=True)

    avg_results = results.groupby(['dataset', 'algorithm'], as_index=False).mean()
    evaluator.set_benchmark_results(avg_results)
    evaluator.export_results('run_real_datasets')
    evaluator.create_boxplots(runs=RUNS, data=results, detectorwise=False)
    evaluator.create_boxplots(runs=RUNS, data=results, detectorwise=True)



if __name__ == '__main__':
    main()

#######################################################################################
import gc
import logging
import os
import pickle
import re
import sys
import traceback
from textwrap import wrap

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
import progressbar
import time
from sklearn.metrics import accuracy_score, fbeta_score
from sklearn.metrics import precision_recall_fscore_support as prf
from sklearn.metrics import roc_curve, auc, confusion_matrix
#from sklearn.metrics import roc_curve, auc
from tabulate import tabulate

from .config import init_logging


class Evaluator:
    def __init__(self, datasets: list, detectors: callable, output_dir: {str} = None, seed: int = None,
                 create_log_file=True):
        """
        :param datasets: list of datasets
        :param detectors: callable that returns list of detectors
        """
        assert np.unique([x.name for x in datasets]).size == len(datasets), 'Some datasets have the same name!'
        self.datasets = datasets
        self._detectors = detectors
        self.output_dir = output_dir or 'reports'
        self.results = dict()
        if create_log_file:
            init_logging(os.path.join(self.output_dir, 'logs'))
        self.logger = logging.getLogger(__name__)
        # Dirty hack: Is set by the main.py to insert results from multiple evaluator runs
        self.benchmark_results = None
        # Last passed seed value in evaluate()
        self.seed = seed

    @property
    def detectors(self):
        detectors = self._detectors(self.seed)
        assert np.unique([x.name for x in detectors]).size == len(detectors), 'Some detectors have the same name!'
        return detectors

    def set_benchmark_results(self, benchmark_result):
        self.benchmark_results = benchmark_result

    def export_results(self, name):
        output_dir = os.path.join(self.output_dir, 'evaluators')
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime('%Y-%m-%d-%H%M%S')
        path = os.path.join(output_dir, f'{name}-{timestamp}.pkl')
        self.logger.info(f'Store evaluator results at {os.path.abspath(path)}')
        save_dict = {
            'datasets': [x.name for x in self.datasets],
            'detectors': [x.name for x in self.detectors],
            'benchmark_results': self.benchmark_results,
            'results': self.results,
            'output_dir': self.output_dir,
            'seed': int(self.seed),
        }
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
            print(save_dict['benchmark_results'].to_string())
        return path

    # Import benchmark_results if this evaluator uses the same detectors and datasets
    # self.results are not available because they are overwritten by each run
    def import_results(self, name):
        output_dir = os.path.join(self.output_dir, 'evaluators')
        path = os.path.join(output_dir, f'{name}.pkl')
        self.logger.info(f'Read evaluator results at {os.path.abspath(path)}')
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)

        self.logger.debug(f'Importing detectors {"; ".join(save_dict["detectors"])}')
        my_detectors = [x.name for x in self.detectors]
        assert np.array_equal(save_dict['detectors'], my_detectors), 'Detectors should be the same'

        self.logger.debug(f'Importing datasets {"; ".join(save_dict["datasets"])}')
        my_datasets = [x.name for x in self.datasets]
        assert np.array_equal(save_dict['datasets'], my_datasets), 'Datasets should be the same'

        self.benchmark_results = save_dict['benchmark_results']
        self.seed = save_dict['seed']
        self.results = save_dict['results']

    @staticmethod
    def get_accuracy_precision_recall_fscore(y_true: list, y_pred: list):
        accuracy = accuracy_score(y_true, y_pred)
        # warn_for=() avoids log warnings for any result being zero
        precision, recall, f_score, _ = prf(y_true, y_pred, average='binary', warn_for=())
        if precision == 0 and recall == 0:
            f01_score = 0
        else:
            f01_score = fbeta_score(y_true, y_pred, average='binary', beta=0.1)
        return accuracy, precision, recall, f_score, f01_score

    @staticmethod
    def get_auroc(det, ds, score):
        if np.isnan(score).all():
            score = np.zeros_like(score)
        _, _, _, y_test = ds.data()
        score_nonan = score.copy()
        # Rank NaN below every other value in terms of anomaly score
        score_nonan[np.isnan(score_nonan)] = np.nanmin(score_nonan) - sys.float_info.epsilon
        fpr, tpr, _ = roc_curve(y_test, score_nonan)
        return auc(fpr, tpr)

    def get_optimal_threshold(self, det, y_test, score, steps=100, return_metrics=False):
        maximum = np.nanmax(score)
        minimum = np.nanmin(score)
        threshold = np.linspace(minimum, maximum, steps)
        metrics = list(self.get_metrics_by_thresholds(y_test, score, threshold))
        metrics = np.array(metrics).T
        anomalies, acc, prec, rec, f_score, f01_score = metrics
        if return_metrics:
            return anomalies, acc, prec, rec, f_score, f01_score, threshold
        else:
            return threshold[np.argmax(f01_score)]

    def evaluate(self):
        for ds in progressbar.progressbar(self.datasets):
            (X_train, y_train, X_test, y_test) = ds.data()
            for det in progressbar.progressbar(self.detectors):
                self.logger.info(f'Training {det.name} on {ds.name} with seed {self.seed}')
                try:
                    det.fit(X_train.copy())
                    score = det.predict(X_test.copy())
                    self.results[(ds.name, det.name)] = score
                    try:
                        self.plot_details(det, ds, score)
                    except Exception:
                        pass
                except Exception as e:
                    self.logger.error(f'An exception occurred while training {det.name} on {ds}: {e}')
                    self.logger.error(traceback.format_exc())
                    self.results[(ds.name, det.name)] = np.zeros_like(y_test)
            gc.collect()

    def benchmarks(self) -> pd.DataFrame:
        df = pd.DataFrame()
        for ds in self.datasets:
            _, _, _, y_test = ds.data()
            for det in self.detectors:
                score = self.results[(ds.name, det.name)]
                y_pred = self.binarize(score, self.get_optimal_threshold(det, y_test, np.array(score)))
                acc, prec, rec, f1_score, f01_score = self.get_accuracy_precision_recall_fscore(y_test, y_pred)
                confusion_mat = confusion_matrix(y_test, y_pred, labels=[0, 1])
                self.plot_confusion_matrix(det, ds, confusion_mat)
                score = self.results[(ds.name, det.name)]
                auroc = self.get_auroc(det, ds, score)
                df = df.append({'dataset': ds.name,
                                'algorithm': det.name,
                                'accuracy': acc,
                                'precision': prec,
                                'recall': rec,
                                'F1-score': f1_score,
                                'F0.1-score': f01_score,
                                'auroc': auroc,
                                'confusion_mat':confusion_mat
                                },
                               ignore_index=True)
        return df

    def plot_confusion_matrix(self, det, ds, conf_matrix, store=True):
                plt.close('all')
                fig, ax = plt.subplots(figsize=(7.5, 7.5))
                ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
                for i in range(conf_matrix.shape[0]):
                    for j in range(conf_matrix.shape[1]):
                        ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

                plt.xlabel('Predictions', fontsize=18)
                plt.ylabel('Actuals', fontsize=18)
                plt.title('Confusion Matrix', fontsize=18)
                if store:
                    self.store(fig, f'confusion_matrix-{det.name}-{ds.name}')
                return fig

    def get_metrics_by_thresholds(self, y_test: list, score: list, thresholds: list):
        for threshold in thresholds:
            anomaly = self.binarize(score, threshold=threshold)
            metrics = Evaluator.get_accuracy_precision_recall_fscore(y_test, anomaly)
            yield (anomaly.sum(), *metrics)

    def plot_scores(self, store=True):
        detectors = self.detectors
        plt.close('all')
        figures = []
        for ds in self.datasets:
            X_train, y_train, X_test, y_test = ds.data()
            subtitle_loc = 'left'
            fig = plt.figure(figsize=(15, 15))
            fig.canvas.set_window_title(ds.name)

            sp = fig.add_subplot((2 * len(detectors) + 3), 1, 1)
            sp.set_title('original training data', loc=subtitle_loc)
            for col in X_train.columns:
                plt.plot(X_train[col])
            sp = fig.add_subplot((2 * len(detectors) + 3), 1, 2)
            sp.set_title('original test set', loc=subtitle_loc)
            for col in X_test.columns:
                plt.plot(X_test[col])

            sp = fig.add_subplot((2 * len(detectors) + 3), 1, 3)
            sp.set_title('binary labels of test set', loc=subtitle_loc)
            plt.plot(y_test)

            subplot_num = 4
            for det in detectors:
                sp = fig.add_subplot((2 * len(detectors) + 3), 1, subplot_num)
                sp.set_title(f'scores of {det.name}', loc=subtitle_loc)
                score = self.results[(ds.name, det.name)]
                plt.plot(np.arange(len(score)), [x for x in score])
                threshold_line = len(score) * [self.get_optimal_threshold(det, y_test, np.array(score))]
                plt.plot([x for x in threshold_line])
                subplot_num += 1

                sp = fig.add_subplot((2 * len(detectors) + 3), 1, subplot_num)
                sp.set_title(f'binary labels of {det.name}', loc=subtitle_loc)
                plt.plot(np.arange(len(score)),
                         [x for x in self.binarize(score, self.get_optimal_threshold(det, y_test, np.array(score)))])
                subplot_num += 1
            fig.subplots_adjust(top=0.9, hspace=0.4)
            fig.tight_layout()
            if store:
                self.store(fig, f'scores_{ds.name}')
            figures.append(fig)
        return figures

    def plot_threshold_comparison(self, steps=40, store=True):
        detectors = self.detectors
        plt.close('all')
        plots_shape = len(detectors), len(self.datasets)
        fig, axes = plt.subplots(*plots_shape, figsize=(len(detectors) * 15, len(self.datasets) * 5))
        # Ensure two dimensions for iteration
        axes = np.array(axes).reshape(*plots_shape).T
        plt.suptitle('Compare thresholds', fontsize=10)
        for ds, axes_row in zip(self.datasets, axes):
            _, _, X_test, y_test = ds.data()

            for det, ax in zip(detectors, axes_row):
                score = np.array(self.results[(ds.name, det.name)])

                anomalies, _, prec, rec, f_score, f01_score, thresh = self.get_optimal_threshold(
                    det, y_test, score, return_metrics=True)

                ax.plot(thresh, anomalies / len(y_test),
                        label=fr'anomalies ({len(y_test)} $\rightarrow$ 1)')
                ax.plot(thresh, prec, label='precision')
                ax.plot(thresh, rec, label='recall')
                ax.plot(thresh, f_score, label='f_score', linestyle='dashed')
                ax.plot(thresh, f01_score, label='f01_score', linestyle='dashed')
                ax.set_title(f'{det.name} on {ds.name}')
                ax.set_xlabel('Threshold')
                ax.legend()

        # Avoid overlapping title and axis labels
        plt.xlim([0.0, 1.0])
        fig.subplots_adjust(top=0.9, hspace=0.4, right=1, left=0)
        fig.tight_layout()
        if store:
            self.store(fig, 'metrics_by_thresholds')
        return fig

    def plot_roc_curves(self, store=True):
        detectors = self.detectors
        plt.close('all')
        figures = []
        for ds in self.datasets:
            _, _, _, y_test = ds.data()
            fig_scale = 3
            fig = plt.figure(figsize=(fig_scale * len(detectors), fig_scale))
            fig.canvas.set_window_title(ds.name + ' ROC')
            fig.suptitle(f'ROC curve on {ds.name}', fontsize=14, y='1.1')
            subplot_count = 1
            for det in detectors:
                self.logger.info(f'Plotting ROC curve for {det.name} on {ds.name}')
                score = self.results[(ds.name, det.name)]
                if np.isnan(score).all():
                    score = np.zeros_like(score)
                # Rank NaN below every other value in terms of anomaly score
                score[np.isnan(score)] = np.nanmin(score) - sys.float_info.epsilon
                fpr, tpr, _ = roc_curve(y_test, score)
                roc_auc = auc(fpr, tpr)
                plt.subplot(1, len(detectors), subplot_count)
                plt.plot(fpr, tpr, color='darkorange',
                         lw=2, label='area = %0.2f' % roc_auc)
                subplot_count += 1
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.gca().set_aspect('equal', adjustable='box')
                plt.title('\n'.join(wrap(det.name, 20)))
                plt.legend(loc='lower right')
            plt.tight_layout()
            if store:
                self.store(fig, f'roc_{ds.name}')
            figures.append(fig)
        return figures

    def plot_auroc(self, store=True, title='AUROC'):
        plt.close('all')
        self.benchmark_results[['dataset', 'algorithm', 'auroc']].pivot(
            index='algorithm', columns='dataset', values='auroc').plot(kind='bar')
        plt.legend(loc=3, framealpha=0.5)
        plt.xticks(rotation=20)
        plt.ylabel('AUC', rotation='horizontal', labelpad=20)
        plt.title(title)
        plt.ylim(ymin=0, ymax=1)
        plt.tight_layout()
        if store:
            self.store(plt.gcf(), 'auroc', store_in_figures=True)

    def plot_details(self, det, ds, score, store=True):
        if not det.details:
            return
        plt.close('all')
        cmap = plt.get_cmap('inferno')
        _, _, X_test, y_test = ds.data()

        grid = 0
        for value in det.prediction_details.values():
            grid += 1 if value.ndim == 1 else value.shape[0]
        grid += X_test.shape[1]  # data
        grid += 1 + 1  # score and gt

        fig, axes = plt.subplots(grid, 1, figsize=(15, 1.5 * grid))

        i = 0
        c = cmap(i / grid)
        axes[i].set_title('test data')
        for col in X_test.values.T:
            axes[i].plot(col, color=c)
            i += 1
        c = cmap(i / grid)

        axes[i].set_title('test gt data')
        axes[i].plot(y_test.values, color=c)
        i += 1
        c = cmap(i / grid)

        axes[i].set_title('scores')
        axes[i].plot(score, color=c)
        i += 1
        c = cmap(i / grid)

        for key, values in det.prediction_details.items():
            axes[i].set_title(key)
            if values.ndim == 1:
                axes[i].plot(values, color=c)
                i += 1
            elif values.ndim == 2:
                for v in values:
                    axes[i].plot(v, color=c)
                    i += 1
            else:
                self.logger.warning('plot_details: not sure what to do')
            c = cmap(i / grid)

        fig.tight_layout()
        if store:
            self.store(fig, f'details_{det.name}_{ds.name}')
        return fig

    # create boxplot diagrams for auc values for each algorithm/dataset per algorithm/dataset
    def create_boxplots(self, runs, data, detectorwise=True, store=True):
        target = 'algorithm' if detectorwise else 'dataset'
        grouped_by = 'dataset' if detectorwise else 'algorithm'
        relevant_results = data[['algorithm', 'dataset', 'auroc']]
        figures = []
        for det_or_ds in (self.detectors if detectorwise else self.datasets):
            relevant_results[relevant_results[target] == det_or_ds.name].boxplot(by=grouped_by, figsize=(15, 15))
            plt.suptitle('')  # boxplot() adds a suptitle
            plt.title(f'AUC grouped by {grouped_by} for {det_or_ds.name} over {runs} runs')
            plt.ylim(ymin=0, ymax=1)
            plt.tight_layout()
            figures.append(plt.gcf())
            if store:
                self.store(plt.gcf(), f'boxplot_auc_for_{det_or_ds.name}_{runs}_runs', store_in_figures=True)
        return figures

    # create bar charts for averaged pipeline results per algorithm/dataset
    def create_bar_charts(self, runs, detectorwise=True, store=True):
        target = 'algorithm' if detectorwise else 'dataset'
        grouped_by = 'dataset' if detectorwise else 'algorithm'
        relevant_results = self.benchmark_results[['algorithm', 'dataset', 'auroc']]
        figures = []
        for det_or_ds in (self.detectors if detectorwise else self.datasets):
            relevant_results[relevant_results[target] == det_or_ds.name].plot(x=grouped_by, kind='bar', figsize=(7, 7))
            plt.suptitle('')  # boxplot() adds a suptitle
            plt.title(f'AUC for {target} {det_or_ds.name} over {runs} runs')
            plt.ylim(ymin=0, ymax=1)
            plt.tight_layout()
            figures.append(plt.gcf())
            if store:
                self.store(plt.gcf(), f'barchart_auc_for_{det_or_ds.name}_{runs}_runs', store_in_figures=True)
        return figures

    def store(self, fig, title, extension='pdf', no_counters=False, store_in_figures=False):
        timestamp = time.strftime('%Y-%m-%d-%H%M%S')
        if store_in_figures:
            output_dir = os.path.join(self.output_dir, 'figures',title)
        else:
            output_dir = os.path.join(self.output_dir, 'figures', f'seed-{self.seed}',title)
        os.makedirs(output_dir, exist_ok=True)
        counters_str = '' if no_counters else f'-{len(self.detectors)}-{len(self.datasets)}'
        path = os.path.join(output_dir, f'{counters_str}-{timestamp}.{extension}')
        fig.savefig(path)
        self.logger.info(f'Stored plot at {path}')

    def store_text(self, content, title, extension='txt'):
        timestamp = int(time.time())
        output_dir = os.path.join(self.output_dir, 'tables', f'seed-{self.seed}')
        path = os.path.join(output_dir, f'{title}-{len(self.detectors)}-{len(self.datasets)}-{timestamp}.{extension}')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(content)
        self.logger.info(f'Stored {extension} file at {path}')

    def print_merged_table_per_dataset(self, results):
        for ds in self.datasets:
            table = tabulate(results[results['dataset'] == ds.name], headers='keys', tablefmt='psql')
            self.logger.info(f'Dataset: {ds.name}\n{table}')

    def gen_merged_latex_per_dataset(self, results, title_suffix=None, store=True):
        title = f'latex_merged{f"_{title_suffix}" if title_suffix else ""}'
        content = ''
        for ds in self.datasets:
            content += f'''{ds.name}:\n\n{tabulate(results[results['dataset'] == ds.name],
                                                   headers='keys', tablefmt='latex')}\n\n'''
        if store:
            self.store_text(content=content, title=title, extension='tex')
        return content

    def print_merged_table_per_algorithm(self, results):
        for det in self.detectors:
            table = tabulate(results[results['algorithm'] == det.name], headers='keys', tablefmt='psql')
            self.logger.info(f'Detector: {det.name}\n{table}')

    def gen_merged_latex_per_algorithm(self, results, title_suffix=None, store=True):
        title = f'latex_merged{f"_{title_suffix}" if title_suffix else ""}'
        content = ''
        for det in self.detectors:
            content += f'''{det.name}:\n\n{tabulate(results[results['algorithm'] == det.name],
                                   headers='keys', tablefmt='latex')}\n\n'''
        if store:
            self.store_text(content=content, title=title, extension='tex')
        return content

    @staticmethod
    def translate_var_key(key_name):
        if key_name == 'pol':
            return 'Pollution'
        if key_name == 'mis':
            return 'Missing'
        if key_name == 'extremeness':
            return 'Extremeness'
        if key_name == 'f':
            return 'Multivariate'
        # self.logger('Unexpected dataset name (unknown variable in name)')
        return None

    @staticmethod
    def get_key_and_value(dataset_name):
        # Extract var name and value from dataset name
        var_re = re.compile(r'.+\((\w*)=(.*)\)')
        # e.g. 'Syn Extreme Outliers (pol=0.1)'
        match = var_re.search(dataset_name)
        if not match:
            # self.logger.warn('Unexpected dataset name (not variable in name)')
            return '-', dataset_name
        var_key = match.group(1)
        var_value = match.group(2)
        return Evaluator.translate_var_key(var_key), var_value

    @staticmethod
    def get_dataset_types(mi_df):
        types = mi_df.index.get_level_values('Type')
        indexes = np.unique(types, return_index=True)[1]
        return [types[index] for index in sorted(indexes)]

    @staticmethod
    def insert_multi_index_yaxis(ax, mi_df):
        type_title_offset = -1.6  # depends on string length of xaxis ticklabels

        datasets = mi_df.index
        dataset_types = Evaluator.get_dataset_types(mi_df)  # Returns unique entries keeping original order
        logging.getLogger(__name__).debug('Plotting heatmap for groups {" ".join(dataset_types)}')

        ax.set_yticks(np.arange(len(datasets)))
        ax.set_yticklabels([x[1] for x in datasets])

        y_axis_title_pos = 0  # Store at which position we are for plotting the next title
        for idx, dataset_type in enumerate(dataset_types):
            section_frame = mi_df.iloc[mi_df.index.get_level_values('Type') == dataset_type]
            # Somehow it's sorted by its occurence (which is what we want here)
            dataset_levels = section_frame.index.remove_unused_levels().levels[1]
            title_pos = y_axis_title_pos + 0.5 * (len(dataset_levels) - 1)
            ax.text(type_title_offset, title_pos, dataset_type, ha='center', va='center', rotation=90,
                    fontproperties=FontProperties(weight='bold'))
            if idx < len(dataset_types) - 1:
                sep_pos = y_axis_title_pos + (len(dataset_levels) - 0.6)
                ax.text(-0.5, sep_pos, '_' * int(type_title_offset * -10), ha='right', va='center')
            y_axis_title_pos += len(dataset_levels)

    @staticmethod
    def to_multi_index_frame(evaluators):
        evaluator = evaluators[0]
        for other_evaluator in evaluators[1:]:
            assert evaluator.detectors == other_evaluator.detectors, 'All evaluators should use the same detectors'
        pivot_benchmarks = [ev.benchmark_results.pivot(index='dataset', columns='algorithm',
                                                       values='auroc') for ev in evaluators]

        concat_benchmarks = pd.concat(pivot_benchmarks)
        auroc_matrix = concat_benchmarks.groupby(['dataset']).mean()

        datasets = [[evaluator.get_key_and_value(str(d)) for d in ev.index.values]
                    for ev in pivot_benchmarks]
        datasets = [tuple(d) for d in np.concatenate(datasets)]  # Required for MultiIndex.from_tuples
        datasets = pd.MultiIndex.from_tuples(datasets, names=['Type', 'Level'])
        auroc_matrix.index = datasets
        return auroc_matrix

    def get_multi_index_dataframe(self):
        return Evaluator.to_multi_index_frame([self])

    @staticmethod
    def plot_heatmap(evaluators, store=True):
        mi_df = Evaluator.to_multi_index_frame(evaluators)
        detectors, datasets = mi_df.columns, mi_df.index

        fig, ax = plt.subplots(figsize=(len(detectors) + 2, len(datasets)))
        im = ax.imshow(mi_df, cmap=plt.get_cmap('YlOrRd'), vmin=0, vmax=1)
        plt.colorbar(im)

        # Show MultiIndex for ordinate
        Evaluator.insert_multi_index_yaxis(ax, mi_df)

        # Rotate the tick labels and set their alignment.
        ax.set_xticks(np.arange(len(detectors)))
        ax.set_xticklabels(detectors)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

        # Loop over data dimensions and create text annotations.
        for i in range(len(detectors)):
            for j in range(len(datasets)):
                ax.text(i, j, f'{mi_df.iloc[j, i]:.2f}', ha='center', va='center', color='w',
                        path_effects=[path_effects.withSimplePatchShadow(
                            offset=(1, -1), shadow_rgbFace='b', alpha=0.9)])

        ax.set_title('AUROC over all datasets and detectors')
        # Prevent bug where x axis ticks are completely outside of bounds (matplotlib/issues/5456)
        if len(datasets) > 2:
            fig.tight_layout()
        if store:
            evaluators[0].store(fig, 'heatmap', no_counters=True, store_in_figures=True)
        return fig

    def plot_single_heatmap(self, store=True):
        Evaluator.plot_heatmap([self], store)

    @staticmethod
    def get_printable_runs_results(results):
        print_order = ['dataset', 'algorithm', 'accuracy', 'precision', 'recall', 'F1-score', 'F0.1-score', 'auroc']
        rename_columns = [col for col in print_order if col not in ['dataset', 'algorithm']]

        # calc std and mean for each algorithm per dataset
        std_results = results.groupby(['dataset', 'algorithm']).std(ddof=0).fillna(0)
        # get rid of multi-index
        std_results = std_results.reset_index()
        std_results = std_results[print_order]
        std_results.rename(inplace=True, index=str,
                           columns=dict([(old_col, old_col + '_std') for old_col in rename_columns]))

        avg_results = results.groupby(['dataset', 'algorithm'], as_index=False).mean()
        avg_results = avg_results[print_order]

        avg_results_renamed = avg_results.rename(
            index=str, columns=dict([(old_col, old_col + '_avg') for old_col in rename_columns]))
        return std_results, avg_results, avg_results_renamed

    def gen_merged_tables(self, results, title_suffix=None, store=True):
        title_suffix = f'_{title_suffix}' if title_suffix else ''
        std_results, avg_results, avg_results_renamed = Evaluator.get_printable_runs_results(results)

        ds_title_suffix = f'per_dataset{title_suffix}'
        self.print_merged_table_per_dataset(std_results)
        self.gen_merged_latex_per_dataset(std_results, f'std_{ds_title_suffix}', store=store)

        self.print_merged_table_per_dataset(avg_results_renamed)
        self.gen_merged_latex_per_dataset(avg_results_renamed, f'avg_{ds_title_suffix}', store=store)

        det_title_suffix = f'per_algorithm{title_suffix}'
        self.print_merged_table_per_algorithm(std_results)

        self.gen_merged_latex_per_algorithm(std_results, f'std_{det_title_suffix}', store=store)
        self.print_merged_table_per_algorithm(avg_results_renamed)
        self.gen_merged_latex_per_algorithm(avg_results_renamed, f'avg_{det_title_suffix}', store=store)

    def binarize(self, score, threshold=None):
        threshold = threshold if threshold is not None else self.threshold(score)
        score = np.where(np.isnan(score), np.nanmin(score) - sys.float_info.epsilon, score)
        return np.where(score >= threshold, 1, 0)

    def threshold(self, score):
        return np.nanmean(score) + 2 * np.nanstd(score)