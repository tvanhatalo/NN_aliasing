import torch
import torch.nn as nn 
import torch.jit as jit
import numpy as np
from tqdm import tqdm
from data_processor import reshape_data 
from utils import ConvBlock, LSTMCell, LSTMLayer, LowPass, LPConvBlock

""" Models """

# WaveNet from https://github.com/Alec-Wright/CoreAudioML - last accessed 11/05/2021

class WaveNet(torch.nn.Module):
    def __init__(self, buffer_size: int = 2048, blocks=2, nb_layers: int = 10, dilation_factor: int = 2, nb_channels:int = 16, kernel_size:int=3, sample_rate:int = 352800, LP_type:str = None):
        super(WaveNet, self).__init__()
        self.nb_layers = nb_layers
        self.dilation = dilation_factor
        self.channels = nb_channels
        self.layers = nb_layers
        self.buffer_size = buffer_size
        self.kernel_size = kernel_size
        self.sr = sample_rate
        self.blocks = nn.ModuleList()
        self.LP_type = LP_type
        if self.LP_type:
            for b in range(blocks):
                self.blocks.append(LPConvBlock(self.dilation, self.layers, 1 if b == 0 else self.channels, self.channels, self.kernel_size, self.sr, self.LP_type))
            self.blocks.append(nn.Conv1d(self.channels*self.layers*blocks, 1, 1, 1, 0))
        else:
            for b in range(blocks):
                self.blocks.append(ConvBlock(self.dilation, self.layers, 1 if b == 0 else self.channels, self.channels, self.kernel_size, self.sr))
            self.blocks.append(nn.Conv1d(self.channels*self.layers*blocks, 1, 1, 1, 0))
    def forward(self, x):
        x = x.permute(1, 2, 0)
        skips = torch.empty([x.shape[0], self.blocks[-1].in_channels, x.shape[2]]).cuda()
        for n, block in enumerate(self.blocks[:-1]):
            x, skip = block(x)
            skips[:, n*self.channels*self.layers:(n + 1) * self.channels*self.layers, :] = skip
        return self.blocks[-1](skips).permute(2, 0, 1)
    def train_loop(self, x_train, y_train,  loss_fn, optimizer, batch_size = 40):
        self.reshaped_x_train = reshape_data(x_train, self.buffer_size) 
        self.reshaped_y_train = reshape_data(y_train, self.buffer_size)
        epoch_loss = 0  
        loss = 0
        nb_batches = int(self.reshaped_x_train.shape[1] // batch_size) + 1
        for i in tqdm(range(nb_batches), desc="Training loop"):
            self.zero_grad()
            x_batch = self.reshaped_x_train[:, i*batch_size:(i+1)*batch_size, :] 
            y_batch = self.reshaped_y_train[:, i*batch_size:(i+1)*batch_size, :] 
            y_pred = self(x_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            self.hidden = None
        return epoch_loss/(i+1)
    def predict(self, x, validation:bool=False, y_true = None, loss_fn = None, buffer_size:int = None): # cuda OOM without buffer processing
        torch.cuda.empty_cache()
        if buffer_size is not None:
            self.buffer_size = buffer_size
        if validation:
            assert y_true is not None 
            assert loss_fn is not None
            assert buffer_size is not None
            batch_size = 40
            x = reshape_data(x, buffer_size).cuda()
            y_true = reshape_data(y_true, buffer_size).cuda()
            epoch_loss = 0  
            loss = 0
            nb_batches = int(x.shape[1] // batch_size) + 1
            for i in tqdm(range(nb_batches), desc="Training loop"):
                self.zero_grad()
                x_batch = x[:, i*batch_size:(i+1)*batch_size, :] 
                y_batch = y_true[:, i*batch_size:(i+1)*batch_size, :] 
                y_pred = self(x_batch)
                loss = loss_fn(y_pred, y_batch).item()
                epoch_loss += loss
            return y_pred, loss
        else:
            with torch.no_grad():
                x = reshape_data(x).cuda()
                assert x.size()[0] != 0
                y_pred = torch.empty_like(x)
                for l in tqdm(range(int(x.size()[0] / self.buffer_size)), desc="Inference progress: "):
                    y_pred[l * self.buffer_size:(l + 1) * self.buffer_size] = self(x[l * self.buffer_size:(l + 1) * self.buffer_size])
                if y_true is not None:
                    y_true = reshape_data(y_true).cuda()
                    assert loss_fn is not None
                    loss = loss_fn(y_pred, y_true)
                    return y_pred, loss
        torch.cuda.empty_cache()
        return y_pred

class LSTM(torch.nn.Module):
    def __init__(self, custom : bool = False, buffer_size: int = 2048, nb_units : int = 32, sample_rate:int = 352800):
        super(LSTM, self).__init__()
        self.buffer_size = buffer_size
        self.nb_units = nb_units
        self.custom = custom
        self.sr = sample_rate
        self.LP = LowPass()
        if self.custom:
            # device = torch.device("cpu")
            num_directions = 1
            h_zeros = torch.zeros(num_directions,
                                  self.nb_units, 
                                  device=torch.device("cuda:0"))
            c_zeros = torch.zeros(num_directions,
                                  self.nb_units,
                                  device=torch.device("cuda:0"))
            self.init_hidden = (h_zeros, c_zeros)
            self.cell = LSTMCell
            self.lstm = LSTMLayer(self.cell, 1, self.nb_units)
        else:
            self.init_hidden = None 
            self.lstm = torch.nn.LSTM(1, self.nb_units, 2, batch_first = False)
        self.hidden = self.init_hidden
        self.dense = torch.nn.Linear(nb_units, 1, bias = True)
    def forward(self, inputs):
        outputs, self.hidden = self.lstm(inputs, self.hidden)
        outputs = self.dense(outputs)
        return outputs
    def detach_hidden(self):
        self.hidden = tuple([h.clone().detach() for h in self.hidden])
    def train_loop(self, x_train, y_train, loss_fn, optimizer, batch_size = 50): 
        self.reshaped_x_train = reshape_data(x_train, self.buffer_size) 
        self.reshaped_y_train = reshape_data(y_train, self.buffer_size)
        loss = 0
        epoch_loss = 0  
        nb_batches = int(self.reshaped_x_train.shape[1] // batch_size) + 1 
        for i in tqdm(range(nb_batches), desc="Training loop"): 
            buffer_size = self.buffer_size
            x_batch = self.reshaped_x_train[:, i*batch_size:(i+1)*batch_size, :] 
            y_batch = self.reshaped_y_train[:, i*batch_size:(i+1)*batch_size, :] 
            batch_loss = 0
            start = 0
            for k in range(int(np.ceil(x_batch.shape[0]/buffer_size))):
                y_pred = self(x_batch[start:start + buffer_size, :, :]) 
                y_true = y_batch[start:start + buffer_size, :, :]
                loss = loss_fn(y_pred, y_true)
                loss.backward()
                optimizer.step()
                self.hidden = tuple([h.clone().detach() for h in self.hidden])
                self.zero_grad()
                start += buffer_size
                batch_loss += loss.item()
            epoch_loss += batch_loss / (k+1)
            self.hidden = self.init_hidden 
        return epoch_loss/(i+1)
    def predict(self, x, validation:bool=False, y_true = None, loss_fn = None, buffer_size: int = 2048): 
        with torch.no_grad():
            x = reshape_data(x).cuda()
            assert x.size()[0] != 0
            y_pred = torch.empty_like(x)
            for l in tqdm(range(int(x.size()[0] / buffer_size)), desc="Inference progress: "):
                y_pred[l * buffer_size:(l + 1) * buffer_size] = self(x[l * buffer_size:(l + 1) * buffer_size])
                self.hidden = tuple([h.clone().detach() for h in self.hidden])
            if not (x.size()[0] / buffer_size).is_integer():
                y_pred[(l + 1) * buffer_size:] = self(x[(l + 1) * buffer_size:])
            self.hidden = self.init_hidden 
            if y_true is not None:
                assert loss_fn is not None
                y_true = reshape_data(y_true).cuda()
                loss = loss_fn(y_pred, y_true).item()
                return y_pred, loss 
        return y_pred
