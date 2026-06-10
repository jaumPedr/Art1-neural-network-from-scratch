import numpy as np
import pandas as pd
import json
from ucimlrepo import fetch_ucirepo 
from tqdm import tqdm

def data_module():
    #get data set
    dry_bean = fetch_ucirepo(id=602) 
                                                   
    #features and targets
    X = dry_bean.data.features 
    y = dry_bean.data.targets


    X_quartiles = []

    #Separete each colum in 4 quartiles
    for col in X.columns:
        bins = pd.qcut(X[col], q=4, duplicates='drop')
        X_quartiles.append(pd.get_dummies(bins, prefix=col))

    #one-hot encoding, each quartiles is a feature (Dummy Variables)
    X_bin = pd.concat(X_quartiles, axis=1)
    X_bin = X_bin.astype(int).to_numpy()
    

    return X_bin,y

class Art():
    def __init__(self, comparison_layer_size, recognition_layer_size, p):
        #F1 layer size
        self.comparison_layer_size = comparison_layer_size
        #F2 layer size
        self.recognition_layer_size = recognition_layer_size

        #vigilance parameter
        self.p = p  

        #1 if node activate, 0 if not
        self.node_status = np.ones(shape = (recognition_layer_size))

        #forward weights
        self.w_forward = np.ones(shape = (recognition_layer_size, comparison_layer_size))
        self.w_forward = self.w_forward * ( 1 / ( 1 +  comparison_layer_size) )

        #backward weights
        self.w_backward = np.ones(shape = (comparison_layer_size, recognition_layer_size))
        
        
    #recognition step
    def forward(self, x):
        
        #forward
        u = self.w_forward @ x
        
        #disable rejected nodes
        u[self.node_status == 0] = -np.inf

        #activation function
        winner_node_index = np.argmax(u)

        return winner_node_index
    
    #comparison step
    def backward(self, x, winner_node_index):
        
        #similarity ratio
        R = ( x @ self.w_backward[:,winner_node_index] ) / ( np.sum(x) )

        return R > self.p

    #weight adaptation
    def weight_update(self, x, winner_node_index):
        
        product = x * self.w_backward[:,winner_node_index]

        #update forward weights
        self.w_forward[winner_node_index, :] = product / ( 0.5 + (x @ self.w_backward[:,winner_node_index]))
        
        #update backward weights
        self.w_backward[:, winner_node_index] =  product

#category search step
def train(model : Art, x):

    #ignore nule class
    if np.all(x == 0):
        return -1

    while (True):
        #select winner node
        k = model.forward(x)
        
        #apply backward function
        vigilance_satisfied = model.backward(x, k)
        
        if (vigilance_satisfied) :
            #learn current pattern
            model.weight_update(x, k)
            
            # reactivate all nodes
            model.node_status[:] = 1
            
            return k
        else:
            #reject current category if vigilance not satisfied
            model.node_status[k] = 0

            #create new category if needed
            if np.all( model.node_status == 0 ):
                add_node(model)
            

#add new category to model
def add_node(model : Art):
    #increase recognition layer size
    model.recognition_layer_size += 1

    #increase node status size
    model.node_status = np.zeros(shape = (model.recognition_layer_size))
    model.node_status[model.recognition_layer_size - 1] = 1
    
    #create new row to forward weight
    new_w_f = np.ones(shape = (model.comparison_layer_size))
    new_w_f = new_w_f * ( 1 / ( 1 +  model.comparison_layer_size) )
    model.w_forward = np.vstack( (model.w_forward, new_w_f) )
    
    #create new column to backward weight
    new_w_b = np.ones(shape = (model.comparison_layer_size))
    model.w_backward = np.column_stack((model.w_backward, new_w_b))


#train all input patterns
def train_loop(model : Art, X):
    K = []
    for x in tqdm(X):
        K.append(train(model, x))
    
    Y = []

    # build one-hot outputs
    for k in K:

        if k == -1:
            y = -1
        else:
            y = np.zeros(shape= (model.recognition_layer_size))
            y[k] = 1
        Y.append(y)

    Y = np.array(Y, dtype=int)

    return Y, K

#ART Test  
X, y = data_module()

model = Art(X.shape[1], 1, p = 0.035)
Y, K = train_loop(model, X)

np.savetxt('./results/art_output_classes.csv', K, delimiter=',', fmt='%d')
np.savetxt('./results/art_output_one_hot_encoding.csv', Y, delimiter=',', fmt='%d')
print('Number of Classes: ', model.recognition_layer_size)

print(y.squeeze)

df = pd.DataFrame({
    'Classe_Real': y.squeeze(),
    'Categoria_ART': K
})

print(pd.crosstab(df['Categoria_ART'], df['Classe_Real']))
