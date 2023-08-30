import torch
import os
import soundfile as sf
import numpy as np
import torch.nn as nn
from typing import Tuple, Optional, List
from torch import Tensor
from torch.autograd import Variable
from torch.fft import fft
from matplotlib import pyplot as plt
import torchaudio
from scipy.signal import kaiserord, butter, firwin, filtfilt, lfilter
from data_processor import reshape_data 
import json

class Arctan(torch.nn.Module):
    """
        Applies arctangent function element-wise (within the bounds [-1, 1]).
        ArctanAct(x) = 2/pi * arctan(pi/2 * x)
    """
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = np.pi/2.0 * x
        y = torch.atan(x) 
        return 2.0/np.pi * y

class LowPass(nn.Module):
    def __init__(self, sample_rate:int=22050):
        super(LowPass, self).__init__()
        self.sample_rate = sample_rate
        self.cut_off = sample_rate*0.5
    def forward(self, inputs):
        outputs = torchaudio.functional.lowpass_biquad(inputs, self.sample_rate, 0.8*self.cut_off)
        return outputs.cuda()

def seed_everything(seed: int):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def generate_test_input(sample_rate : int):
    T = 1/sample_rate
    t = 10 # in seconds
    N = sample_rate * t
    f_0 = 1244.5
    omega = 2*np.pi*f_0 
    time = np.arange(N)*T
    out = np.sin(omega*time) 
    sf.write(os.path.join("data", f"test_sine_{sample_rate}.wav"), out, sample_rate)        

def plot_mag_spec(file_name:str="sine_output_22k"):
    assert os.path.isfile(os.path.join("data", f"{file_name}.wav")), "Input wav file should be in data folder."
    test, test_fs = sf.read(os.path.join("data", f"{file_name}.wav"))
    time_test = np.arange(len(test))
    plt.plot(time_test[:1000], test[:1000])
    plt.show()
    fig, axs = plt.subplots()
    axs.set_title("Magnitude Spectrum")
    axs.magnitude_spectrum(test, Fs=test_fs, scale='dB')
    axs.set_xlim([0, 22050])
    axs.set_ylim([-115, 0])
    axs.set_xlabel("Frequency (Hz)")
    plt.show()

""" Losses """

def STFT_mag(x, n_fft:int, hop_length:int=None, win_length:int=None, window=None):
    x = torch.transpose(x[:,:,0], 0, 1)
    stft = torch.stft(x, n_fft, hop_length, win_length, window, return_complex=True)
    # Re = stft[:, :, :, 0]
    # Im = stft[:, :, :, 1]
    return torch.abs(stft)

class STFTLoss(nn.Module):
    """ 
    STFT loss used from [Yamamoto et al. 2020] https://arxiv.org/pdf/1910.11480.pdf.
    """
    def __init__(self, n_fft:int, hop_length:int=None, win_length:int=None, stft_factor:float=0.1, eps:float=0.0000001):
        super(STFTLoss, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.stft_factor = stft_factor
        self.eps = eps
        self.MAE = nn.L1Loss()
    def SM(self, stft_mag_pred, stft_mag_true):
        return 1/self.n_fft * self.MAE(torch.log(stft_mag_true + self.eps), torch.log(stft_mag_pred + self.eps))
    def SC(self, stft_mag_pred, stft_mag_true):
        return torch.norm(stft_mag_true - stft_mag_pred, p="fro") / torch.norm(stft_mag_true, p="fro")
    def forward(self, y_pred, y_true):
        stft_mag_pred = STFT_mag(y_pred, self.n_fft, self.hop_length, self.win_length) 
        stft_mag_true = STFT_mag(y_true, self.n_fft, self.hop_length, self.win_length)
        return self.MAE(y_pred, y_true) + self.stft_factor*(self.SM(stft_mag_pred, stft_mag_true) + self.SC(stft_mag_pred, stft_mag_true))

class MultiResSTFTLoss(nn.Module):
    def __init__(self, n_filters=None, windows_size=None, hops_size=None):
        super(MultiResSTFTLoss, self).__init__()
        if n_filters is None:
            n_filters = [2048, 1024, 512, 256, 128, 64, 32]
        if windows_size is None:
            windows_size = [2048, 1024, 512, 256, 128, 64, 32]
        if hops_size is None:
            hops_size = [1024, 512, 256, 128, 64, 32, 16]
        self.n_filters = n_filters
        self.windows_size = windows_size
        self.hops_size = hops_size
        self.losses = nn.ModuleList()
        for f, h, w in zip(self.n_filters, self.windows_size, self.hops_size):
            self.losses += [STFTLoss(f, h, w)]

    def forward(self, y_pred, y_true):
        loss = 0.0
        for l in self.losses:
            loss += l(y_pred, y_true)
        return loss/len(self.losses)
    
""" WaveNet """
    
# WaveNet blocks from https://github.com/Alec-Wright/CoreAudioML - last accessed 11/05/2021
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, kernel_size, sample_rate:int = None):
        super(ConvLayer, self).__init__()
        self.channels = out_channels
        self.dilated_conv = nn.Conv1d(in_channels=in_channels, out_channels=2*out_channels, kernel_size=kernel_size, stride=1, padding = 0, dilation = dilation) 
        self.out_conv = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.arctan = Arctan()
        self.hardtanh = torch.nn.Hardtanh(-0.95,0.95) 
    def forward(self, x):
        residual = x
        x = self.dilated_conv(x)
        # skips = self.arctan(x[:, :self.channels, :]) * torch.sigmoid(x[:,self.channels:, :])
        skips = torch.tanh(x[:, :self.channels, :]) * torch.sigmoid(x[:,self.channels:, :])
        skips = torch.cat((torch.zeros(residual.shape[0], self.channels, residual.shape[2]-skips.shape[2]).cuda(), skips), dim=2)
        x = self.out_conv(skips) + residual 
        return x, skips

class LPConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, kernel_size, sample_rate, LP_type:str="AF_GAN"):
        super(LPConvLayer, self).__init__()
        self.channels = out_channels
        self.sr = sample_rate
        self.dilation = dilation
        self.dilated_conv = nn.Conv1d(in_channels=in_channels, out_channels=2*out_channels, kernel_size=kernel_size, stride=1, padding = 0, dilation = dilation) 
        self.dense_conv = nn.Conv1d(in_channels=in_channels, out_channels=2*out_channels, kernel_size=kernel_size, stride=1, padding = 0, dilation = 1) 
        self.out_conv = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.low_pass = LowPass()
        self.LP_type = LP_type
    def forward(self, x):
        residual = x
        if self.LP_type == "BlurPooling":
            # BlurPooling:
            # dense conv -> activation -> LP filter -> downsampling
            x = self.dense_conv(x)
            # activation
            skips = torch.tanh(x[:, :self.channels, :]) * torch.sigmoid(x[:,self.channels:, :])
            skips = torch.cat((torch.zeros(residual.shape[0], self.channels, residual.shape[2]-skips.shape[2]).cuda(), skips), dim=2)
            # lp filter
            skips = self.low_pass(skips)
            # downsample/dilate
            x = x[:,:,::self.dilation]
            # skips = skips[:,:,::self.dilation]
            x = self.out_conv(skips) + residual 
        elif self.LP_type == "PostFilter":
            # Vasconcelos a.k.a. PostFilter:
            # dense conv -> LP filter -> downsample -> activation
            x = self.dense_conv(x)
            # lp filter
            x = self.low_pass(x)
            # dilation
            x = x[:,:,::self.dilation]
            # activation 
            skips = torch.tanh(x[:, :self.channels, :]) * torch.sigmoid(x[:,self.channels:, :])
            skips = torch.cat((torch.zeros(residual.shape[0], self.channels, residual.shape[2]-skips.shape[2]).cuda(), skips), dim=2)
            x = self.out_conv(skips) + residual 
        elif self.LP_type == "AF_GAN":
            # oversampling the nonlinearities - by a factor of 2 was deem sufficient for equivariance
            x = self.dilated_conv(x)
            # oversample the activation
            x = nn.functional.interpolate(x, scale_factor=2)
            residual = nn.functional.interpolate(residual, scale_factor=2)
            skips = torch.tanh(x[:, :self.channels, :]) * torch.sigmoid(x[:,self.channels:, :])
            skips = torch.cat((torch.zeros(residual.shape[0], self.channels, residual.shape[2]-skips.shape[2]).cuda(), skips), dim=2)
            x = self.out_conv(skips) + residual 
            x = nn.functional.interpolate(x, scale_factor=1/2)
            skips = nn.functional.interpolate(skips, scale_factor=1/2)
        else: 
            assert False, "LP type not known."
        return x, skips

class ConvBlock(nn.Module):
    def __init__(self, dilation_factor, nb_layers, in_channels, out_channels, kernel_size, sample_rate):
        super(ConvBlock, self).__init__()
        self.channels = out_channels
        dilations = [dilation_factor ** k for k in range(nb_layers)]
        self.layers = nn.ModuleList()
        for d in dilations:
            self.layers.append(ConvLayer(in_channels, out_channels, d, kernel_size, sample_rate))
            in_channels = out_channels
    def forward(self, x):
        skips = torch.empty(x.shape[0], len(self.layers)*self.channels, x.shape[2])
        for n, layer in enumerate(self.layers):
            x, skip = layer(x)
            skips[:, n*self.channels: (n+1)*self.channels, :] = skip
        return x, skips
    
class LPConvBlock(nn.Module):
    def __init__(self, dilation_factor, nb_layers, in_channels, out_channels, kernel_size, sample_rate, LP_type="AF_GAN"):
        super(LPConvBlock, self).__init__()
        self.channels = out_channels
        dilations = [dilation_factor ** k for k in range(nb_layers)]
        self.layers = nn.ModuleList()
        for d in dilations:
            self.layers.append(LPConvLayer(in_channels, out_channels, d, kernel_size, sample_rate, LP_type = LP_type))
            in_channels = out_channels
    def forward(self, x):
        skips = torch.empty(x.shape[0], len(self.layers)*self.channels, x.shape[2])
        for n, layer in enumerate(self.layers):
            x, skip = layer(x)
            skips[:, n*self.channels: (n+1)*self.channels, :] = skip
        return x, skips

""" LSTM """

def get_lstm_weights(nb_units : int):
    # Make sure to set seed 
    lstm_for_weights = torch.nn.LSTM(1, nb_units, 1, batch_first = False)
    w_ih = lstm_for_weights.weight_ih_l0
    w_hh = lstm_for_weights.weight_hh_l0
    b_ih = lstm_for_weights.bias_ih_l0
    b_hh = lstm_for_weights.bias_hh_l0
    return w_ih, w_hh, b_ih, b_hh

class LSTMCell(torch.jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # initialise like standard lstm 
        self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh = get_lstm_weights(hidden_size)
        self.arctan = Arctan()
        self.hardtanh = torch.nn.Hardtanh(-0.95,0.95) 
    @torch.jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        # original activations
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        # cellgate = self.arctan(cellgate) 
        cellgate = torch.tanh(cellgate) # replace with tanh variants
        outgate = torch.sigmoid(outgate)
        cy = (forgetgate * cx) + (ingate * cellgate)
        # hy = outgate * self.arctan(cy)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)

class LSTMLayer(torch.jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super(LSTMLayer, self).__init__()
        self.cell = cell(*cell_args)
    @torch.jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        inputs = input.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state
