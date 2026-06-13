import numpy as np
from tqdm import tqdm

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
        if np.sum(x) == 0:
            return False
        
        #similarity ratio
        R = ( x @ self.w_backward[:,winner_node_index] ) / ( np.sum(x) )

        return R >= self.p

    #weight adaptation
    def weight_update(self, x, winner_node_index):
        
        product = x * self.w_backward[:,winner_node_index]

        #update forward weights
        self.w_forward[winner_node_index, :] = product / ( 0.5 + (x @ self.w_backward[:,winner_node_index]))
        
        #update backward weights
        self.w_backward[:, winner_node_index] =  product

#category search step
def fit(model : Art, features, max_recognitio_layer_size = np.inf):

    #ignore nule class
    if np.all(features == 0):
        return -1

    while (True):
        #select winner node
        k = model.forward(features)
        
        #apply backward function
        vigilance_satisfied = model.backward(features, k)
        
        if (vigilance_satisfied) :
            #learn current pattern
            model.weight_update(features, k)
            
            # reactivate all nodes
            model.node_status[:] = 1
            return k
        
        else:
            #reject current category if vigilance not satisfied
            model.node_status[k] = 0

            #create new category if needed
            if np.all( model.node_status == 0 ):
                if model.recognition_layer_size >= max_recognitio_layer_size:
                    model.node_status[:] = 1
                    return -1
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

def predict(model : Art, X):
    Y = []
    K = []

    for x in X:

        model.node_status[:] = 1

        if np.all(x == 0):
            K.append(-1)
            continue

        while True:

            k = model.forward(x)

            if model.backward(x, k):
                K.append(k)
                break

            model.node_status[k] = 0

            if np.all(model.node_status == 0):
                K.append(-1)
                break

    for k in K:

        if k == -1:
            y = -1
        else:
            y = np.zeros(shape= (model.recognition_layer_size))
            y[k] = 1
        Y.append(y)
    #Y = np.array(Y, dtype=int)

    return Y, K


#train all input patterns
def train_loop(model : Art, X, epochs = 5):
    K = []
    for e in range(epochs):
        for x in tqdm(X):
            fit(model, x)

    for x in tqdm(X):
            K.append(fit(model, x))

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


