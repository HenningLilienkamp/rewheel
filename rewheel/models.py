from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rewheel.helper_utils import prev_pow_2

from torch.utils.data import DataLoader, TensorDataset

def get_layer_config(input_size, output_size):
    N_nodes = [input_size]
    N = input_size
    while N > output_size:
        N = 2**(prev_pow_2(N))
        N_nodes.append(N)
        N -= 1
    N_nodes[-1] = output_size

    return N_nodes

def accuracy(y_hat, y):
    out = ((y_hat.round() == y).sum(axis=1) == y.shape[1]).sum() / \
        len(y)
    return out

class MLP_core(nn.Module):
    def __init__(self, n_nodes):
        super(MLP_core, self).__init__()
        # initialize the layers
        for i_layer in range(len(n_nodes)-1):
            setattr(self,
                    'layer%i'%(i_layer+1),
                    nn.Linear(n_nodes[i_layer],
                              n_nodes[i_layer+1]))
    
        # initialize activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        layer_names = [entry for entry in dir(self) if 
                       entry.startswith('layer')]
        # define the forward pass
        l_in = getattr(self, layer_names[0])
        x = l_in(x)
        for layer_name in layer_names[1:]:
            # print(layer_name)
            x = self.relu(x)
            l = getattr(self, layer_name)
            # print(l)
            x = l(x)
        x = self.softmax(x)
        return x

class AbstractModel(ABC):
    @abstractmethod
    def teach(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def export(self):
        pass

class MLP(AbstractModel):
    def __init__(self,
                 input_shape=None,
                 output_shape=None,
                 model_file=None):
        
        self.__model_type = 'MLP'

        if not (model_file or (input_shape and output_shape)):
            raise AssertionError('Cannot create model. \
                                 Either torch model file or I/O \
                                 dimensions required')
        
        if model_file:
            self.model = torch.load(model_file)
            self.in_shape = self.model.layer1.in_features
            layer_names = [entry for entry in dir(self.model) if 
                       entry.startswith('layer')]
            self.out_shape = getattr(self.model, layer_names[-1]).out_features
            self.node_config = [getattr(self.model, name).in_features for name in layer_names] + [getattr(self.model, layer_names[-1]).out_features]


        else:
        
            self.in_shape = np.prod(input_shape)
            self.out_shape = output_shape[0]
            self.node_config = get_layer_config(self.in_shape,
                                                self.out_shape)
            self.model = MLP_core(self.node_config)
        
    def teach(self, X_train, y_train, X_test, y_test, BATCH_SIZE=64,
              N_epochs=10, out_dir='./'):
        
        device = torch.device('cuda' if torch.cuda.is_available() else
                              'cpu')
        X_train = X_train.reshape((len(X_train), self.in_shape))
        X_test = X_test.reshape((len(X_test), self.in_shape))

        X_train = torch.tensor(X_train, dtype=torch.float).to(device)
        print(X_train.shape)
        y_train = torch.tensor(y_train, dtype=torch.float).to(device)
        X_test = torch.tensor(X_test, dtype=torch.float).to(device)
        y_test = torch.tensor(y_test, dtype=torch.float).to(device)

        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True)
        
        model = self.model.to(device)
        loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        accuracy_temp = 0
        for epoch in range(N_epochs):
            for id_batch, (x_batch, y_batch) in enumerate(loader):
                predictions = model(x_batch) 
                CEL = loss(predictions, y_batch)
                CEL.backward()
                optimizer.step()
                optimizer.zero_grad()
                
            training_predictions = model(X_train)
            test_predictions = model(X_test)
            
            CEL_train = loss(training_predictions, y_train)
            CEL_test = loss(test_predictions, y_test)

            accuracy_test = accuracy(test_predictions, y_test)
            print(f'Epoch [{epoch + 1}], CE train Loss: {CEL_train.item()}, CE test Loss: {CEL_test.item()}, test accuracy: {accuracy_test}')
            if accuracy_test > accuracy_temp:
                torch.save(model, out_dir + 'MNIST_MLP_model.pth')
                print('Model improved - Saved to file.')
                accuracy_temp = accuracy_test

        self.model = model
    
    def predict(self, X):
        device = torch.device('cuda' if torch.cuda.is_available() else
                              'cpu')
        X = X.reshape((len(X), self.in_shape))
        X = torch.tensor(X, dtype=torch.float).to(device)
        y_hat = self.model(X).cpu().detach().numpy()
        return y_hat

    def export(self):
        pass
        




