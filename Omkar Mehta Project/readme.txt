# Analysis and Forecasting of Precipitation ERA5 Land Data of Mahad

The file structure is as follows:
- data_collection.ipynb
- milestone2.ipynb
- neural_network.ipynb
- univariate_lstm/
    - train.ipynb
    - utils.py

Following topics are covered in great detail in the following notebooks:
- data_collection.ipynb
    : This notebook is used to collect the data and create a NetCDF file.
    : It uses CDS API client to collect the data.
    : omehta2 Project Milestone 1 Spring 2022.pdf has detailed procedure.

- milestone2.ipynb
    : Science objective, 
    : Data (access, description, statistics, etc.),
    : Exploratory Data Analysis (EDA),
    : Data Cleaning.
    : Data processing like masking, sub-grouping etc.
    : Subsetting of the data for Mahad and Mahabaleshwar
    : Mahad and Mahabaleshwar's comparison results.

- neural_network.ipynb
    : This notebook is used to train the Linear Regression model.
    : Data Processing like smoothing, normalization, interpolation of missing values, etc.
    : Linear Regression model using sm's OLS.
    : Splitting the data into training and test sets.
    : Standardization of the data.
    : Deep Neural Network model and training.

- univariate_lstm/train.ipynb
    : This notebook is used to train advanced models like RNN, LSTM, Attentional LSTM, etc.
    : Mahad's data is used to train the model.
    : Training and results.
    : Model Statistics on Test Data 
    : Inferring from plots
    : Future Work

- univariate_lstm/utils.py
    : This file contains the utility functions used in the notebook.
    : make_dirs(path) is used to create directories.
    : load_data(data) is used to load the data.
    : split_sequence_uni_step() is used to get sequences for single step prediction.
    : split_sequence_multi_step() is used to get sequences for multi step prediction.
    : data_loader() is used to load the data in pytorch.
    : plot_pred_test() is used to plot the test predictions.

- univariate_lstm/models.py
    : This file contains the models used in the notebook.
    : class DNN is used to train the DNN model.
    : class LSTM is used to train the LSTM model.
    : class RNN is used to train the RNN model.
    : class AttentionalLSTM is used to train the Attentional LSTM model.
    : class CNN is used to train the CNN model.

- Download data from https://uofi.box.com/s/7vhifnk9cx2mh54ope73ambmi55e1opw 

