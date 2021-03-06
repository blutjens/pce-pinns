"""
Neural network
Author: Björn Lütjens (lutjens@mit.edu)
"""

import numpy as np
import pandas as pd
#import seaborn as sns
from tqdm import tqdm

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import src.utils.plotting as plotting
import src.rom.pce as pce

class RegressionDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

def init_dataloader(X, y, batch_size, test_size=0.1, val_size=0.1, shuffle=False):
    """
    Args:
        shuffle bool: If True, shuffles dataset. The option is not true, s.t., the rand instance in train and test can be correlated
    """
    if test_size > 0 and val_size > 0:
        # Train - Test
        X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=60)
        # Split train into train-val
        X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=val_size, stratify=y_trainval, random_state=20)

        val_dataset = RegressionDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
        test_dataset = RegressionDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())

        val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=shuffle)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=shuffle)
    else:
        X_train = X
        y_train = y
        val_loader = None
        test_loader = None

    train_dataset = RegressionDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, val_loader, test_loader

class FCNN(nn.Module):
    # https://stackoverflow.com/questions/46141690/how-to-write-a-pytorch-sequential-model
    def __init__(self, n_features, n_out):
        super(FCNN, self).__init__()
        
        n_layers = 2
        n_units = 128 # number of hidden units
        self.act = nn.ReLU#ReLU

        layers = []
        # Input layer
        layers.append(nn.Linear(n_features, n_units))
        layers.append(self.act())
        # Hidden layers
        for l in range(n_layers):
            layers.append(nn.Linear(n_units, n_units))
            layers.append(self.act())
        # Output layer
        layers.append(nn.Linear(n_units, n_out))
        self.net = nn.Sequential(*layers)

    def predict(self, test_inputs):
        x = self.net(test_inputs)
        return (x)

    def forward(self, inputs):
        x = self.net(inputs)
        return (x)

class MSELoss_k_pce(object):
    def __init__(self, alpha_indices,verbose=False, 
            omit_const_pce_coef=False,rand_insts=None):
        """
        Args:
            alpha_indices np.array(pce_dim, n_alpha_indices)
            verbose bool: If true, created verbose prints
            omit_const_pce_coef bool: If true, removes PCE coefficients of zero-th order.
            rand_insts np.array(n_grid*n_samples, n_stoch_dim): Random instances, corresponding to the training set; (opt)
        """
        self.alpha_indices=alpha_indices
        self.verbose=verbose
        if omit_const_pce_coef:
            return NotImplementedError('omitting zero-th order PCE coefs in MSELoss_k_pce not implemented.')
        self.rand_insts = rand_insts
        self.rand_inst = None # Current random instance
        self.iter = 0
    def __call__(self, pce_coefs, k_target):
        """
        Args:
            pce_coefs np.array(n_grid, n_alpha_indices)
            k_target np.array(n_grid, 1)
        """
        # !!!TODO!!! rand_inst need to be shape of (n_batches, batch_size, ndim) 
        k_pred = pce.sample_pce_torch(pce_coefs, self.alpha_indices, rand_inst=self.rand_inst)
        #k_pred = torch.mean(pce_coefs,axis=1)
        mse_loss_k_pce = torch.mean((k_pred - k_target)**2)
        if self.verbose: print('k: mse, pred, target', mse_loss_k_pce, k_pred, k_target)
        #_assert_no_grad(k_target)
        self.iter += 1
        return mse_loss_k_pce # F.mse_loss(k_pred, k_target, size_average=self.size_average, reduce=self.reduce)

    def set_rand_inst(self, batch_id, batch_size):
        start_id = batch_id * batch_size
        end_id = batch_id * batch_size + batch_size
        # rand_inst np.array(batch_size, ndim)
        self.rand_inst = self.rand_insts[start_id:end_id,:]

def train(model, train_loader, optimizer, criterion, n_epochs, device, loss_stats, plot=False, 
    custom_rand_inst=False, batch_size=None):
    """
    Args:
        custom_rand_inst bool: If true, the random instances of the NN-based model and training data are the same   
        batch_size int: Batch size; only used to set rand_inst
    """
    print("Begin training.")
    for e in tqdm(range(1, n_epochs+1)):
        
        # TRAINING
        train_epoch_loss = 0
        
        model.train()
        i = 0
        for X_train_batch, y_train_batch in train_loader:
            #print('i-th batch:', i, y_train_batch[0])
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            # Forward pass: Compute predicted y by passing x to the model    
            y_train_pred = model(X_train_batch)
            
            # Get rand instance id of test batch. TODO: make this independent of n_grid
            if custom_rand_inst:
                criterion.set_rand_inst(batch_id=i, batch_size=batch_size)
            # Compute loss
            loss = criterion(y_train_pred, y_train_batch)#.unsqueeze(1))
           
            # Zero gradients, perform a backward pass, and update the weights. 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_epoch_loss += loss.item()
            
            i += 1
        # TODO: validation
        
        loss_stats['train'].append(train_epoch_loss/len(train_loader))
        
        print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.6f}')

    if plot: 
        plotting.plot_train_curve(loss_stats)

    return model, loss_stats

def predict(model, x_test, device, target_name='pce_coefs', 
    alpha_indices=None, n_test_samples=1, plot=False):
    """
    Args:
        x_test np.array((n_grid,1)): 

    Returns:
        y_test np.array((n_grid, n_alpha_indices))
        k_samples np.array((n_grid, n_samples))
    """
    k_samples = None
    with torch.no_grad():
        model.eval()
        x_test_batch = torch.from_numpy(x_test).to(device).type(torch.float32)
        y_test = model.predict(x_test_batch)
    if plot: 
        # Plot pce_coefs
        plotting.plot_nn_pred(x=x_test_batch.cpu().numpy(), y=y_test.cpu().numpy())
    if target_name == 'k' or target_name=='k_true':
        # Sample param k, given learned deterministic pce_coefs
        k_samples = torch.zeros((n_test_samples, x_test.shape[0]))
        for n in range(n_test_samples):
            k_samples[n,:] = pce.sample_pce_torch(y_test, alpha_indices)[:,0]
        k_samples = k_samples.cpu().numpy()
        if plot:
            plotting.plot_nn_k_samples(x_test[:,0], k_samples)
    return y_test.cpu().numpy(), k_samples

def interpolate_param_nn(xgrid, y,
        n_epochs=30, batch_size=30,
        lr=0.001, 
        target_name='pce_coefs', 
        alpha_indices=None, n_test_samples=1,
        plot=False, rand_insts=None):
    """
    Train interpolating NN that predicts params, PCE coefs, or param eigenvecs 
    Args:
        xgrid np.array(n_samples, n_grid,): Location
        y np.array(n_samples, n_grid, n_out): Target, e.g., pce_coefs or k_target; n_out is stochastic dim, e.g., n_alpha_indices
        target_name (string): Indication which target
        rand_insts np.array((n_samples, n_out)): Random instances, used to generated training dataset

    Returns:
        y_test np.array(n_grid,n_out)
    """
    # init cpu/gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using ', device)

    # TODO. the current reshaping assumes samples are independent in space!!
    if target_name=='pce_coefs':
        # Repeat target into batches for deterministic target
        # TODO: check if this is actually necessary
        repeats = batch_size
        x = np.repeat(xgrid[:, np.newaxis], repeats=repeats, axis=0)
        y = np.repeat(y[:,:], repeats=repeats, axis=0)
        n_out = y.shape[-1] 
    elif target_name=='k' or target_name=='k_true':
        # Merge n_samples and n_grid axes into one. Assumes that samples are iid. across location, x
        x = xgrid.reshape(-1, 1)
        y = y.reshape(-1, 1) 
        n_out = alpha_indices.shape[0]
        # Align the random instances with their samples
        if rand_insts is not None:
            rand_insts = np.repeat(rand_insts[:,:],repeats=xgrid.shape[1],axis=0)
    elif target_name=='mu_k' or 'k_eigvecs':
        # Merge n_samples and n_grid axes into one. Assumes that samples are iid. across location, x
        x = xgrid.reshape(-1, 1)
        n_out = y.shape[-1] 
        y = y.reshape(-1, n_out) 
        #import pdb;pdb.set_trace()
        #plotting.plot_mu_k_vs_ks_nn(x[:,0], y[:,0], 3*np.ones(xgrid.shape[1]))


    n_epochs = n_epochs
    batch_size = batch_size # 64
    lr = lr# 0.001
    n_features = x.shape[-1]

    train_loader, val_loader, test_loader = init_dataloader(x, y, batch_size, test_size=0., val_size=0.)

    model = FCNN(n_features, n_out=n_out)
    model.to(device)
    print(model)

    if target_name == 'pce_coefs' or target_name == 'mu_k' or target_name == 'k_eigvecs':
        criterion = nn.MSELoss()
    elif target_name == 'k' or target_name=='k_true':
        criterion = MSELoss_k_pce(alpha_indices=alpha_indices, verbose=False, rand_insts=rand_insts)#, test_return_pce_coefs=False)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_stats = {
        'train': [],
        "val": []
    }

    model, loss_stats = train(model, train_loader, optimizer, criterion, n_epochs, device, 
        loss_stats, plot=plot, custom_rand_inst=(rand_insts is not None), batch_size=batch_size)

    if target_name=='pce_coefs':
        x_test = xgrid[:,np.newaxis]
    elif target_name=='k' or target_name=='k_true':
        x_test = xgrid[0,:,np.newaxis]
    elif target_name=='mu_k' or target_name=='k_eigvecs':
        x_test = xgrid.reshape(-1, 1)
    y_test, k_samples = predict(model, x_test, device=device, 
        target_name=target_name, alpha_indices=alpha_indices, 
        n_test_samples=n_test_samples, plot=plot)

    return y_test, k_samples
