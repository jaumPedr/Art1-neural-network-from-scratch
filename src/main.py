import art_model as art
import data_module as data
import numpy as np
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def run_model(X,y):
    X_train, y_train, X_validation, y_validation, X_test, y_test = data.train_test_validation_split(X,y)

    # Train
    model = art.Art(X_train.shape[1], 1, p=0.5)
    Y_train, K_train = art.train_loop(model, X_train)

    # Validation
    Y_validation, K_validation = art.predict(model, X_validation)

    # Test
    Y_test, K_test = art.predict(model, X_test)

    print('\nTRAIN: ')
    print('Number of Classes: ', model.recognition_layer_size)
    print('Number of Features: ',X[1].size)

    df = pd.DataFrame({
        'Classe_Real': y_train.squeeze(),
        'Categoria_ART': K_train
    })

    print(pd.crosstab(df['Categoria_ART'], df['Classe_Real']))

    print('\nVALIDATION: ')

    df_validation = pd.DataFrame({
        'Classe_Real': y_validation.squeeze(),
        'Categoria_ART': K_validation
    })

    print(pd.crosstab(
        df_validation['Categoria_ART'],
        df_validation['Classe_Real']
    ))

    # Test Crosstab
    print('\nTEST: ')

    df_test = pd.DataFrame({
        'Classe_Real': y_test.squeeze(),
        'Categoria_ART': K_test
    })

    print(pd.crosstab(
        df_test['Categoria_ART'],
        df_test['Classe_Real']
    ))


#ART Test  
X, y = data.get_data()

#binary thresholding
binary_threshold_features = data.binary_thresholding(X)
binary_threshold_features = data.complement_coding(binary_threshold_features)
run_model(binary_threshold_features,y)

#quartiles binary binning
quartiles_binary_binning_features = data.quartiles_binary_binning(X, q=4)
quartiles_binary_binning_features = data.complement_coding(quartiles_binary_binning_features)
run_model(quartiles_binary_binning_features,y)

#thermometer encoding 
thermometer_encoding_features = data.thermometer_encoding(X)
thermometer_encoding_features = data.complement_coding(thermometer_encoding_features)
run_model(thermometer_encoding_features,y)

