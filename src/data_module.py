import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42


def get_data():
    #get dataset
    dry_bean = fetch_ucirepo(id=602) 
                                                   
    #features and targets
    features = dry_bean.data.features 
    targets = dry_bean.data.targets
    return features, targets

def train_test_validation_split(features, targets, test_split=0.2, validation_split=0.2):
    #Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=test_split, random_state=RANDOM_SEED)
    #Train-validation split
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_split, random_state=RANDOM_SEED)

    return X_train, y_train, X_validation, y_validation, X_test, y_test

#For each binary feature x, its complement (1 - x) is appended to the feature vector
def complement_coding(binary_features):

    complement_features = 1 - binary_features
    return np.hstack( [binary_features, complement_features] )


def thermometer_encoding(features, levels=12):

    X_thermometer = []

    for col in features.columns:
        #Equal-frequency discretization
        bins = pd.qcut( features[col], q=levels, labels=False, duplicates='drop')

        # Encoded representation
        encoded_col = np.zeros( (len(features), levels), dtype=int)

        #Activate all positions up to the assigned level
        for i, level in enumerate(bins):
            encoded_col[i, :level+1] = 1

        X_thermometer.append(encoded_col)

    return np.hstack(X_thermometer)

def quartiles_binary_binning(features, q=4):
    
    X_quartiles = []

    #Separete each colum in quartiles
    for col in features.columns:
        bins = pd.qcut(features[col], q=q, duplicates='drop')
        X_quartiles.append(pd.get_dummies(bins, prefix=col))

    #one-hot encoding, each quartiles is a feature (Dummy Variables)
    X_bin = pd.concat(X_quartiles, axis=1)
    X_bin = X_bin.astype(int).to_numpy()

    return X_bin 

def binary_thresholding(features):
    
    #Standardize features
    scaler = StandardScaler()
    scaler.fit(features)
    standardized_features = scaler.transform(features)

    # Compute the threshold (median)
    medians_threshold = np.median(standardized_features, axis=1, keepdims=True)

    #Convert to binary representation
    binary_features = []
    binary_features = ( standardized_features >= medians_threshold ).astype(int)

    return binary_features
    
