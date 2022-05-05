import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
from matplotlib import pyplot as plt # plotting

import torch # torch
from torch.utils.data import TensorDataset, DataLoader # import TensorDataset and DataLoader

from sklearn.model_selection import train_test_split # train/test split

def make_dirs(path):
    # make directories if path does not exist
    if not os.path.exists(path):
        os.makedirs(path)


def load_data(data):
    # Load Data
    data_dir = os.path.join(data) # data directory

    data = pd.read_csv(data_dir,
                       # infer_datetime_format=True,
                       parse_dates=['time']
                       )

    data.index = data['time'] # set index to time
    data = data.drop('time', axis=1) # drop time column
    data = data.drop('')

    return data


def plot_full(path, data, feature):
    """Plot Full Graph of Mahad Dataset"""
    data.plot(y=feature, figsize=(16, 8))
    # plot the x label
    plt.xlabel('DateTime', fontsize=10)
    plt.xticks(rotation=45)
    # plot the y label
    plt.ylabel(feature, fontsize=10)
    plt.grid()
    # set the title
    plt.title('{} Total Precipitation'.format(feature))
    # save the figure
    plt.savefig(os.path.join(path, '{} Total Precipitation.png'.format(feature)))
    plt.show()


def split_sequence_uni_step(sequence, n_steps):
    '''
    Rolling Window Function for Univariate

    Parameters
    ----------
    sequence : array
        sequence of data
    n_steps : int
        number of steps

    Returns
    -------
    X : array
        input data
    y : array
        output data
    '''

    # if we want to predict the next n_steps, we need to shift the sequence
    X, y = list(), list()

    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps


        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        
        # get the sequence for the next n_steps for x and y
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]

        # store in X and y. Append method adds to the end of the list
        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)

def split_sequence_multi_step(sequence, n_steps_in, n_steps_out):
    """
    Rolling Window Function for Multi-step
    
    Parameters
    ----------
    sequence : array
        sequence of data
    n_steps_in : int
        number of steps for input
    n_steps_out : int
        number of steps for output
        
    Returns
    -------
    X : array
        input data
    y : array
        output data
    """
    X, y = list(), list()

    for i in range(len(sequence)):
        # find the end of input pattern
        end_ix = i + n_steps_in
        # find the end of the output pattern
        out_end_ix = end_ix + n_steps_out

        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        
        # get the sequence for the next n_steps for x and y
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]

        X.append(seq_x) # input
        y.append(seq_y) # output

    return np.array(X), np.array(y)[:, :, 0] # because we only want the last feature as the output


def data_loader(x, y, train_split, test_split, batch_size):
    """Prepare data by applying sliding windows and return data loader"""

    # Split to Train, Validation and Test Set #
    train_seq, test_seq, train_label, test_label = train_test_split(x, y, train_size=train_split, shuffle=False)
    val_seq, test_seq, val_label, test_label = train_test_split(test_seq, test_label, train_size=test_split, shuffle=False)

    # Convert to Tensor #
    train_set = TensorDataset(torch.from_numpy(train_seq), torch.from_numpy(train_label))
    val_set = TensorDataset(torch.from_numpy(val_seq), torch.from_numpy(val_label))
    test_set = TensorDataset(torch.from_numpy(test_seq), torch.from_numpy(test_label))

    # Data Loader #
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def get_lr_scheduler(lr_scheduler, optimizer):
    """Learning Rate Scheduler"""
    if lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) # step_size = 10, gamma = 0.5
    elif lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5) # mode = 'min', factor = 0.2, threshold = 0.01, patience = 5
    elif lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0) # T_max = 10, eta_min = 0
    else:
        raise NotImplementedError
    return scheduler


def percentage_error(actual, predicted):
    """Percentage Error"""
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res


def mean_percentage_error(y_true, y_pred):
    """Mean Percentage Error"""
    mpe = np.mean(percentage_error(np.asarray(y_true), np.asarray(y_pred))) * 100
    return mpe


def mean_absolute_percentage_error(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    mape = np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred)))) * 100
    return mape


def plot_pred_test(pred, actual, path, feature, model, step):
    """Plot Test set Prediction"""
    plt.figure(figsize=(10, 8))

    plt.plot(pred, label='Pred')
    plt.plot(actual, label='Actual')

    plt.xlabel('Time', fontsize=18)
    plt.ylabel('{}'.format(feature), fontsize=18)

    plt.legend(loc='best')
    plt.grid()

    plt.title('{} Total Precipitation using {} and {}'.format(feature, model.__class__.__name__, step), fontsize=18)
    plt.savefig(os.path.join(path, '{} using {} and {}.png'.format(feature, model.__class__.__name__, step)))