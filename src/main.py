import art_model as art
import data_module as data
import numpy as np
import pandas as pd


#ART Test  
X, y = data.get_data()
X = data.quartiles_dummy_variables(X, q=2)

X_train, y_train, X_validation, y_validation, X_test, y_test = data.train_test_validation_split(X,y)

model = art.Art(X_train.shape[1], 1, p = 0.75)
Y, K = art.train_loop(model, X_train)

np.savetxt('./results/art_output_classes.csv', K, delimiter=',', fmt='%d')
np.savetxt('./results/art_output_one_hot_encoding.csv', Y, delimiter=',', fmt='%d')
print('Number of Classes: ', model.recognition_layer_size)
print('Number of Features: ',X[1].size)
print(y.squeeze)

df = pd.DataFrame({
    'Classe_Real': y_train.squeeze(),
    'Categoria_ART': K
})

print(pd.crosstab(df['Categoria_ART'], df['Classe_Real']))

Y, K = art.predict(model, X_validation)

df = pd.DataFrame({
    'Classe_Real': y_validation.squeeze(),
    'Categoria_ART': K
})

print(pd.crosstab(df['Categoria_ART'], df['Classe_Real']))