import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo 

RANDOM_SEED = 42
TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.2


def get_data():
    #get data set
    dry_bean = fetch_ucirepo(id=602) 
                                                   
    #features and targets
    features = dry_bean.data.features 
    targets = dry_bean.data.targets
    return features, targets

def train_test_validation_split(features, targets):
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=TEST_SPLIT, random_state=RANDOM_SEED)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED)

    return X_train, y_train, X_validation, y_validation, X_test, y_test

def quartiles_dummy_variables(features, q=4):
    
    X_quartiles = []

    #Separete each colum in 4 quartiles
    for col in features.columns:
        bins = pd.qcut(features[col], q=q, duplicates='drop')
        X_quartiles.append(pd.get_dummies(bins, prefix=col))

    #one-hot encoding, each quartiles is a feature (Dummy Variables)
    X_bin = pd.concat(X_quartiles, axis=1)
    X_bin = X_bin.astype(int).to_numpy()

    return X_bin