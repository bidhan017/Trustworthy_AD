# **Interpretable Anomaly Detection with LSTM/GRU Autoencoders and Prototype Layer**
This project implements an **interpretable anomaly detection framework** for time series using LSTM/GRU-based autoencoders combined with a **prototype layer**, allowing anomaly predictions that are both accurate, transparent, and explainable.

## **Features**
- **Sequence to Sequence Modeling:** LSTM-based encoder-decoder autoencoder reconstructs time series input.

- **Prototype Layer:** Learnable prototypes in the hidden space, used to match sequences and explain model predictions.

- **Gaussian Error Modeling:** Anomalies are scored via the negative log-likelihood of reconstruction errors under a multivariate Gaussian, fitted on normal validation error.

- **Interpretability:** Enables inspection of which learned prototype was used for each prediction, and the corresponding typical patterns.

- **Fully Customizable:** Sequence length, number of prototypes, layers, and other hyperparameters can be easily tuned.

## **Architecture Overview**
- **Encoder:** LSTM (or GRU) networks that map input time series windows to latent representations.

- **Decoder:** Reconstructs original sequence from latent code.

- **Prototype Layer:** Learns k prototypes in the latent space; loss functions encourage each input sequence to be close to a prototype, and each prototype to be distinct.

- **Anomaly Scoring:** Negative log-likelihood of reconstruction error under multivariate Gaussian.
