from collections import Counter, OrderedDict
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class MLPTorch(nn.Module):

    def __init__(self, sizes, lr, nbr_epochs, nbr_nonincreasing_epochs, device,
            weight_decay=0, batch_size=64, verbose=False):
        super(MLPTorch, self).__init__()
        self.verbose = verbose
        self.sizes = sizes
        if len(sizes) == 3:
            self.layers = nn.Sequential(
                nn.Linear(sizes[0], sizes[1]),
                nn.ReLU(),
                nn.Linear(sizes[1], sizes[2]),
            )
        elif len(sizes) == 4:
            self.layers = nn.Sequential(
                nn.Linear(sizes[0], sizes[1]),
                nn.ReLU(),
                nn.Linear(sizes[1], sizes[2]),
                nn.ReLU(),
                nn.Linear(sizes[2], sizes[3]),
            )
        else:
            raise ValueError(f'ERROR: Invalid layer sizes {sizes}')
        self.softmax = nn.Softmax(dim=1)
            
        self.lr = lr
        self.nbr_epochs = nbr_epochs
        self.nbr_nonincreasing_epochs = nbr_nonincreasing_epochs
        self.device = device
        self.layers = self.layers.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,
                weight_decay=weight_decay)
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        self.batch_size = batch_size
            
           
    def to_cpu(self):
        self.device = 'cpu'
        self.layers = self.layers.to(self.device)
        self.loss_fn = self.loss_fn.to(self.device)


    def forward(self, features):                               
        return self.layers(features)

    
    def fit(self, X, y):
        #lr, nbr_epochs, nbr_nonincreasing_epochs, device = args
        X, y = self.to_tensor(X), self.to_tensor(y, to_float=False)
        # If the dataset is too large (as is typically the case for the 
        # meta-model training dataset) and if no mini-batching is used,
        # move it to the GPU memory. This speeds up things but consumes
        # more memory.
        if self.batch_size == -1 or len(X) <= 3000:
            on_device = True
            X, y = X.to(self.device), y.to(self.device)
        else: 
            on_device = False

        assert X.size(1) == self.sizes[0], f'{X.size(1)}, {self.sizes[0]}'
        nbr_train = int(len(X) * 0.9)
        #print(nbr_train)
        X_train, y_train, X_val, y_val = X[:nbr_train], y[:nbr_train], \
                X[nbr_train:], y[nbr_train:]

        best_acc = 0
        best_model = None
        nbr_nonincreasing_count = 0
        batch_size = self.batch_size if self.batch_size > 0 else len(X_train)
        for e in range(self.nbr_epochs):
            self.train()

            # Shuffle the data at the beginning of the epoch.
            shuffled_idxs = torch.randperm(len(X_train)).to(self.device)
            X_train, y_train = X_train[shuffled_idxs], y_train[shuffled_idxs]
            for b in range(0, len(X_train), batch_size):
                b_end = min(b+batch_size, len(X_train))
                X_batch = X_train[b:b_end]
                y_batch = y_train[b:b_end]

                if not on_device:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                y_pred = self.forward(X_batch)
                loss = self.loss_fn(y_pred, y_batch)
                
                if self.verbose:
                    print(f'Epoch {e} [{b_end}/{len(X_train)}. Batch loss: {loss.item():.2f}')
                loss.backward()
                self.optimizer.step()

            self.eval()
            with torch.no_grad():
                if not on_device:
                    X_val_device, y_val_device = X_val.to(self.device), \
                        y_val.to(self.device)
                else:
                    X_val_device, y_val_device = X_val, y_val

                y_pred = self.forward(X_val_device)
                y_pred = torch.argmax(y_pred, dim=1)
            acc_eval = torch.sum(y_pred == y_val_device) / len(y_val_device)
            if self.verbose:
                print(f'End of epoch {e}, {acc_eval.item():.2%}')
            if acc_eval > best_acc:
                best_acc = acc_eval
                best_model = copy.deepcopy(self)
                nbr_nonincreasing_count = 0
            else:
                nbr_nonincreasing_count += 1
                if nbr_nonincreasing_count == self.nbr_nonincreasing_epochs:
                    if self.verbose:
                        print(f'Stopping the training after {e} epochs')
                    break
        return best_model
    
    
    def predict_proba(self, X):
        X = self.to_tensor(X).to(self.device)
        with torch.no_grad():
            return self.softmax(self.forward(X)).cpu().numpy()
    
    
    def predict(self, X):
        with torch.no_grad():
            proba = self.predict_proba(X)
            return np.argmax(proba, axis=1)
        

    @torch.no_grad()
    def get_weights(self):
        weights = []
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                weights.append(layer.weight.view(-1).cpu().numpy())
                weights.append(layer.bias.view(-1).cpu().numpy())
        return np.concatenate(weights)


    @torch.no_grad()
    def get_weights_canonical(self):
        weights = []
        biases = []
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                weights.append(layer.weight.cpu().numpy())
                biases.append(layer.bias.cpu().numpy())
        canonical_weights = []
        for i in range(len(weights)-1):
            sort_idxs = np.argsort(np.sum(weights[i], axis=1))
            weights[i] = weights[i][sort_idxs].flatten()
            biases[i] = biases[i][sort_idxs].flatten()
            canonical_weights.append(weights[i])
            canonical_weights.append(biases[i])
            weights[i+1] = weights[i+1][:, sort_idxs]
        canonical_weights.append(weights[-1].flatten())
        canonical_weights.append(biases[-1].flatten())
        return np.concatenate(canonical_weights)


    @torch.no_grad()
    def sort_weights_canonical(self):
        weights = []
        biases = []
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                weights.append(layer.weight.cpu().numpy())
                biases.append(layer.bias.cpu().numpy())
        for i in range(len(weights)-1):
            sort_idxs = np.argsort(np.sum(weights[i], axis=1))
            weights[i] = weights[i][sort_idxs]
            biases[i] = biases[i][sort_idxs]
            weights[i+1] = weights[i+1][:, sort_idxs]
        i = 0
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                layer.weight = torch.nn.Parameter(
                        torch.FloatTensor(weights[i]).to(self.device))
                layer.bias = torch.nn.Parameter(
                        torch.FloatTensor(biases[i]).to(self.device))
                i += 1

        
    def to_tensor(self, data, to_float=True):
        if not isinstance(data, np.ndarray):
            data = data.to_numpy()
        if to_float:
            data = torch.FloatTensor(data)
        else:
            data = torch.LongTensor(data)
        return data

