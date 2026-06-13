import art_model as art
import data_module as data
import numpy as np
import pandas as pd

#ART Test  
X, y = data.get_data()
X_train, y_train, X_test, y_test, X_validation, y_validation = data.train_test_validation_split(X,y)

X = data.quartiles_dummy_variables(X)

model = art.Art(X.shape[1], 1, p = 0.5)
Y, K = art.train_loop(model, X)

np.savetxt('./results/art_output_classes.csv', K, delimiter=',', fmt='%d')
np.savetxt('./results/art_output_one_hot_encoding.csv', Y, delimiter=',', fmt='%d')
print('Number of Classes: ', model.recognition_layer_size)

print(y.squeeze)

df = pd.DataFrame({
    'Classe_Real': y.squeeze(),
    'Categoria_ART': K
})

print(pd.crosstab(df['Categoria_ART'], df['Classe_Real']))