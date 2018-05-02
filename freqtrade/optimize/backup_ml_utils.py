""" ML utils for data collection and crypto price prediction."""
# import _pickle as cPickle
# import numpy as np
import pandas as pd
# import datetime
# import gdax, time
# from sklearn import preprocessing
# from datetime import datetime
# import operator
# from pandas_datareader import data, wb
# import re
# from dateutil import parser
# from pandas_datareader_gdax import get_data_gdax
from sklearn.ensemble import GradientBoostingClassifier
""" These imports are not used right now.
from backtest import Strategy, Portfolio
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
"""

# def run_pipeline(coins_list, out_coin, start_time, end_time, granularity, test_ratio):
#     """Main util function that runs full pipeline.

#        Returns dataframe with test set and up/down forecasts from machine
#        learning model on test. out_coin is the coin that we want to preeict
#        and itmust not be in the coins_list.
#     """
#     data = DownloadData(coins_list, granularity, start_time, end_time)
#     print(data)
#     filter_col = [col for col in data if col.startswith(out_coin)]
#     out_coin_prices = data[filter_col]
#     X_train, y_train, X_test, y_test = PreprocessData(out_coin, data, test_ratio)
#     parameters = []
#     savemodel = False
#     (accuracy, predictions) = performGTBClass(X_train, y_train, X_test, y_test, parameters, out_coin, savemodel)
#     print ("Prediction accuracy on test set: ", accuracy)
#     # Predictions is pd series of predicted Up/downs indexed by date.
#     output = pd.concat([out_coin_prices, predictions], axis = 1)
#     output[out_coin+'_Prediction'] = output[out_coin+'_Prediction'].shift(1)
#     # NaN values correspond to training set, they should be dropped.
#     output = output.dropna()
#     return output

def run_pipeline(data, out_coin, test_ratio):
    """Main util function that runs full pipeline.

       Returns dataframe with test set and up/down forecasts from machine
       learning model on test. out_coin is the coin that we want to preeict
       and itmust not be in the coins_list.
    """
    filter_col = [col for col in data if col.startswith(out_coin)]
  
    out_coin_prices = data[filter_col]

    X_train, y_train, X_test, y_test = PreprocessData(out_coin, data, test_ratio)

    parameters = []
    savemodel = False
    (accuracy, predictions) = performGTBClass(X_train, y_train, X_test, y_test, parameters, out_coin, savemodel)
    print ("Prediction accuracy on test set: ", accuracy)
    # Predictions is pd series of predicted Up/downs indexed by date.
    output = pd.concat([out_coin_prices, predictions], axis = 1)
    output[out_coin+'_Prediction'] = output[out_coin+'_Prediction'].shift(1)
    # NaN values correspond to training set, they should be dropped.
    output = output.dropna()
    return output

def DownloadData(coins_list, granularity, start_time, end_time):
    merged = pd.DataFrame()
    for coin in coins_list:
      dat = getCoin(coin ,granularity, start_time, end_time)
      dat.columns = [coin+'_Low', coin+'_High', coin+'_Open', coin+'_Close', coin+'_Volume']
      merged = pd.concat([merged, dat], axis=1)
    return merged

def PreprocessData(out_coin, df, test_ratio):
    """ Data preprocessing standard for ML.

    out_coin: symbol of the coin that we want to forecast
    df is a dataframe of all input coins available.
    """
    # apply time lag to the variable we want to forecast
    df[out_coin+'_close'] = df[out_coin+'_close'].shift(2)
    
    #d all variables converted to percent changes

    df = df.pct_change()

    df['UpDown'] = df[out_coin+'_close']
    del df[out_coin+'_close']
    def binarize(x):
      if x>=0:
        return 1
      else:
        return 0
    df['UpDown'] = df['UpDown'].apply(binarize)

    y = df.UpDown
    X = df.copy()
    del X['UpDown']
    X = X.interpolate()
    X = X.fillna(0)
    y = y.interpolate()
    y = y.fillna(0)
    
    num_testing_points = int(len(X.index)*test_ratio)
    print ("Number of testing points: ", num_testing_points)
    X_train = X[num_testing_points:]
    y_train = y[num_testing_points:]
    X_test = X[2:num_testing_points]
    y_test = y[2:num_testing_points]
    print ("Mean of y for training set: ", y_train.mean())
    print ("Mean of y for testing set: ", y_test.mean())
    return X_train, y_train, X_test, y_test

def getCoin(symbol, granularity, start_time, end_time):
    """Downloads Stock from GDAX.

       Returns pandas dataframe.
    """
    df = get_data_gdax(symbol, granularity, start_time, end_time)
    return df

def getCoinsFromWeb(coins_list, fout, start_time, end_time, granularity):
    """Collects predictors data from GDAX.

       Returns a list of dataframes.
       fout is the series that we want to forecast.
    """
    #TODO: add optionality to save data to disk.
    output = []
    for coin in coins_list:
       output.append(getCoin(coin, granularity, start_time, end_time))
    return output

def applyTimeLag(dataset,fout):
    """Applies time lag to series of returns.

     So that explanatory variables dont contain future information.
    """
    dataset['Return_'+fout] = dataset['Return_'+fout].shift(5)
    lagged_data = dataset.iloc[5:,:]
    return lagged_data

def performClassification(X_train, y_train, X_test, y_test, method, parameters, fout, savemodel):
    """Performs classification on daily returns using several algorithms.
    method --> string algorithm
    parameters --> list of parameters passed to the classifier (if any)
    fout --> string with name of stock to be predicted
    savemodel --> boolean. If TRUE saves the model to pickle file
    """
    #TODO: Implement all the methods below, so far we use only Gradient b
    # oosting and I believe it's the best shot to start
    if method == 'RF':
        return performRFClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel)

    elif method == 'KNN':
        return performKNNClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel)

    elif method == 'SVM':
        return performSVMClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel)

    elif method == 'ADA':
        return performAdaBoostClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel)

    elif method == 'GTB':
        return performGTBClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel)

    elif method == 'QDA':
        return performQDAClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel)

def performGTBClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel):
    """
    Gradient Tree Boosting binary Classification
    """
    print ("GTBrunning")
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate = 0.01, max_depth = 5)
    clf.fit(X_train, y_train)

    if savemodel == True:
        fname_out = '{}-{}.pickle'.format(fout, datetime.now())
        with open(fname_out, 'wb') as f:
            cPickle.dump(clf, f, -1)
    prediction = clf.predict(X_test)
    accuracy = clf.score(X_test, y_test)
    prediction_indexed = pd.DataFrame()
    prediction_indexed[fout+'_Prediction'] = pd.Series(prediction, index = X_test.index)
    return (accuracy, prediction_indexed)

def count_missing(df):
    return df.isnull().sum()

def performFeatureSelection(maxdeltas, maxlags, fout, cut, start_test, path_datasets, savemodel, method, folds, parameters):
    """Performs Feature selection for a specific algorithm.
    """
    #TODO: Implement rigorous feature seleciton. We don't need it for now.
    for maxlag in range(3, maxlags + 2):
        lags = range(2, maxlag)
        print ('')
        print ('=============================================================')
        print ('Maximum time lag applied', max(lags))
        print ('')
        for maxdelta in range(3, maxdeltas + 2):
            datasets = loadDatasets(path_datasets)
            delta = range(2, maxdelta)
            print ('Delta days accounted: ', max(delta))
            datasets = applyRollMeanDelayedReturns(datasets, delta)
            finance = mergeDataframes(datasets, 6, cut)
            print ('Size of data frame: ', finance.shape)
            print ('Number of NaN after merging: ', count_missing(finance))
            finance = finance.interpolate(method='linear')
            print ('Number of NaN after time interpolation: ', count_missing(finance))
            finance = finance.fillna(finance.mean())
            print ('Number of NaN after mean interpolation: ', count_missing(finance))
            #TODO: get time lag correctly
            #finance = applyTimeLag(finance, lags, delta)
            print ('Number of NaN after temporal shifting: ', count_missing(finance))
            print ('Size of data frame after feature creation: ', finance.shape)
            X_train, y_train, X_test, y_test  = prepareDataForClassification(finance, start_test,fout)
            print(y_train,"y_train")
            clf = GradientBoostingClassifier(n_estimators=100)
            clf.fit(X_train, y_train)
            predictions = clf.predict(X_test)
            print(predictions,"predictions")
            accuracy = clf.score(X_test,y_test)
            print(accuracy,"accuracy")
            #print performTimeSeriesCV(X_train, y_train, folds, method, parameters)
            print ('')
