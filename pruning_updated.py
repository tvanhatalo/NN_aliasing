import torch
import soundfile as sf 
import torch.nn as nn 
import os
import numpy as np
from matplotlib import pyplot as plt
from time import time
import utils
import models
import torch.nn.utils.prune as prune

def model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) 

def load_model(model_type:str, model_file:str, model_size:int, nb_layers:int=None, nb_channels:int=None):
    torch.cuda.set_device(0)
    assert model_type in ["LSTM", "WaveNet"], "Unknown model type."
    if model_type == "WaveNet":
        model = models.WaveNet(buffer_size=22050*10, dilation_factor=model_size, nb_layers=nb_layers, nb_channels=nb_channels, sample_rate=22050).cuda()
    else: 
        model = models.LSTM(custom = False, nb_units=model_size, sample_rate=22050).cuda()
    chck_pt = torch.load(model_file)
    model.load_state_dict(chck_pt) 
    return model

def prune_lstm():
    """ 
        LSTM model pruning 
    """
    for seed in np.arange(0, 100, int(100/25)):
        unpruned_model_file = os.path.join("artefacts", "saved_models", "sparsity_LSTM_105U", f"pytorch_LSTM_50epochs_antialiased_22050_original_input_{seed}.pth")
        dense_model_file = os.path.join("artefacts", "saved_models", "sparsity_LSTM_32U", f"pytorch_LSTM_50epochs_antialiased_22050_original_input_{seed}.pth")
        torch.cuda.set_device(0)

        unpruned_model = load_model("LSTM", unpruned_model_file, 105)
        unpruned_model = unpruned_model.cuda()
        unpruned_model_size = model_size(unpruned_model)

        dense_model = load_model("LSTM", dense_model_file, 32)
        dense_model = dense_model.cuda()
        dense_model_size = model_size(dense_model)

        sample_rate = 22050

        test_input = sf.read(os.path.join("data", "test", "antialiased_22050", f"test_sine_input_{sample_rate}.wav"), dtype='float32')[0]
        test_target = sf.read(os.path.join("data", "test", "antialiased_22050", f"output_sine_test_max_timestep_705600_anti_aliased_{sample_rate}.wav"), dtype='float32')[0]

        loss_fn = nn.L1Loss()

        unpruned_prediction, unpruned_mae = unpruned_model.predict(test_input[:sample_rate*10], y_true=test_target[:sample_rate*10], loss_fn=loss_fn)
        dense_prediction, dense_mae = dense_model.predict(test_input[:sample_rate*10], y_true=test_target[:sample_rate*10], loss_fn=loss_fn)

        module = unpruned_model.lstm
        start = time()
        prune.l1_unstructured(unpruned_model.lstm, name="weight_ih_l0", amount=0.9)
        prune.l1_unstructured(unpruned_model.lstm, name="weight_hh_l0", amount=0.9)
        prune.l1_unstructured(unpruned_model.lstm, name="bias_ih_l0", amount=0.9)
        prune.l1_unstructured(unpruned_model.lstm, name="bias_hh_l0", amount=0.9)
        prune.l1_unstructured(unpruned_model.dense, name="weight", amount=0.9)
        prune.l1_unstructured(unpruned_model.dense, name="bias", amount=0.9)
        duration = time() - start

        print(f"Pruning phase lasted {duration} seconds.")

        prune.remove(unpruned_model.lstm, name="weight_ih_l0")
        prune.remove(unpruned_model.lstm, name="weight_hh_l0")
        prune.remove(unpruned_model.lstm, name="bias_ih_l0")
        prune.remove(unpruned_model.lstm, name="bias_hh_l0")
        prune.remove(unpruned_model.dense, name="weight")
        prune.remove(unpruned_model.dense, name="bias")

        pruned_prediction, pruned_mae = unpruned_model.predict(test_input[:sample_rate*10], y_true=test_target[:sample_rate*10], loss_fn=loss_fn)
        print(f"\nSEED = {seed}\n")
        print(f"unpruned_mae = {unpruned_mae}")
        print(f"dense_mae = {dense_mae}")
        print(f"pruned_mae = {pruned_mae}\n")
        pruned_model_size = model_size(unpruned_model)

def prune_wavenet():
    """ 
        WaveNet model pruning 
    """
    for seed in np.arange(0, 100, int(100/25)):
        unpruned_model_file = os.path.join("artefacts", "saved_models", "sparsity_WaveNet_L10_C5", f"pytorch_WaveNet_50epochs_antialiased_22050_original_input_{seed}.pth")
        dense_model_file = os.path.join("artefacts", "saved_models", "sparsity_WaveNet_L1_C5", f"pytorch_WaveNet_50epochs_antialiased_22050_original_input_{seed}.pth")
        torch.cuda.set_device(0)

        unpruned_model = load_model("WaveNet", unpruned_model_file, dilation=2, nb_layers=10, nb_channels=5)
        unpruned_model = unpruned_model.cuda()
        unpruned_model_size = model_size(unpruned_model)

        dense_model = load_model("WaveNet", dense_model_file, dilation=2, nb_layers=1, nb_channels=5)
        dense_model = dense_model.cuda()
        dense_model_size = model_size(dense_model)

        sample_rate = 22050

        test_input = sf.read(os.path.join("data",  "test", "antialiased_22050", f"test_sine_input_{sample_rate}.wav"), dtype='float32')[0]
        test_target = sf.read(os.path.join("data", "test", "antialiased_22050", f"output_sine_test_max_timestep_705600_anti_aliased_{sample_rate}.wav"), dtype='float32')[0]

        loss_fn = nn.L1Loss()

        unpruned_pred, unpruned_mae = unpruned_model.predict(test_input, y_true=test_target, loss_fn=loss_fn)
        dense_pred, dense_mae = dense_model.predict(test_input, y_true=test_target, loss_fn=loss_fn)
        start = time()
        for name, module in unpruned_model.named_modules():
            if isinstance(module, torch.nn.Conv1d):
                prune.l1_unstructured(module=module, name='weight', amount=0.9)
                prune.l1_unstructured(module=module, name='bias', amount=0.9)
                
        duration = time() - start
        print(f"Pruning phase lasted {duration} seconds.")

        for name, module in unpruned_model.named_modules():
            if isinstance(module, torch.nn.Conv1d):
                prune.remove(module, name='weight')
                prune.remove(module, name='bias')

        # test pruned model
        pruned_pred, pruned_mae = unpruned_model.predict(test_input[:sample_rate*10], False, y_true=test_target[:sample_rate*10], loss_fn=loss_fn)
        print(f"\nSEED = {seed}\n")
        print(f"unpruned_mae = {unpruned_mae}")
        print(f"dense_mae = {dense_mae}")
        print(f"pruned_mae = {pruned_mae}\n")
        pruned_model_size = model_size(unpruned_model)

prune_lstm()
