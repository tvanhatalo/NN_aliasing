import torch
import soundfile as sf 
import torch.nn as nn 
import os
import numpy as np
from matplotlib import pyplot as plt
from time import time
import utils
import models
import csv


""" Main """

def main(model_name, seed:int=42):
    start = time()
    utils.seed_everything(seed)
    sample_rate = 22050
    exp_name = "AF_GAN_factor8"
    if model_name == "LSTM":
        network = models.LSTM(nb_units = 32, custom = False, sample_rate = sample_rate)
    elif model_name == "CustomLSTM":
        network = models.LSTM(custom = True, sample_rate = sample_rate)
    elif model_name == "WaveNet":
        network = models.WaveNet(buffer_size = 2048, nb_layers = 10, dilation_factor = 2, nb_channels = 16, kernel_size = 3, sample_rate = sample_rate, LP_type="AF_GAN") 
    else:
        raise NotImplementedError("This model doesn't exist.")
    loss_fn = nn.L1Loss()
    
    optimiser = torch.optim.Adam(network.parameters(), lr=0.005, weight_decay=1e-4) # lr = 0.005

    # Check CUDA
    if not torch.cuda.is_available():
        print(f'Training {model_name} on CPU')
    elif model_name == "CustomLSTM":
        print("Custom Recurrent Cell: Testing GPU")
        # os.environ["CUDA_VISIBLE_DEVICES"]=""
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.set_device(0)
        print(f'Training {model_name} on GPU')
        network = network.cuda()
    else:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.set_device(0)
        print(f'Training {model_name} on GPU')
        network = network.cuda()

    # check model size 
    param_size = 0
    for param in network.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in network.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all = (param_size + buffer_size) 
    print(f'model size: {size_all}')

    # anamoly detection
    torch.autograd.set_detect_anomaly(True)
    # load data 
    amp = "antialiased_22050" 
    data_path = os.path.join("data")
    input_data, inp_sr =  sf.read(os.path.join(data_path, "train", amp, "input_22050.wav"))
    assert sample_rate == inp_sr
    input_data = np.array(input_data).astype(np.float32, order='C')
    target_data = np.array(sf.read(os.path.join(data_path, "train", amp, "output_train_705600_anti_aliased_22050.wav"))[0]).astype(np.float32, order='C') # output_train_705600_anti_aliased_22050
    # split
    ratio = 0.8
    assert len(input_data) == len(target_data)
    x_train = input_data[:int(ratio*len(input_data))]
    x_val = x_train[-int((1-ratio)*len(x_train)):]
    x_train = x_train[:int(ratio*len(x_train))]
    y_train = target_data[:int(ratio*len(target_data))]
    y_val = y_train[-int((1-ratio)*len(y_train)):]
    y_train = y_train[:int(ratio*len(y_train))]

    nb_epochs = 1

    train_losses = []
    val_losses = [] 
    for epoch in range(nb_epochs):
        epoch_loss = network.train_loop(x_train, y_train, loss_fn, optimiser)
        # validation loop
        if (epoch % 2 == 0): 
            val_pred, val_loss = network.predict(x_val, True, y_val, loss_fn, 2048)
            val_losses.append(val_loss)
        print(f"EPOCH {epoch}: Training loss = {epoch_loss} - Validation loss = {val_loss}.")
        train_losses.append(epoch_loss) #
    
    # save model
    if not os.path.exists(os.path.join("artefacts", "saved_models", exp_name)):
        os.makedirs(os.path.join("artefacts", "saved_models", exp_name))
    torch.save(network.state_dict(), os.path.join("artefacts", "saved_models", exp_name, f"pytorch_{model_name}_{nb_epochs}epochs_{amp}_{seed}.pth"))

    epochs = np.arange(nb_epochs)
    val_epochs = np.arange(stop=nb_epochs, step=2)
    plt.plot(epochs, train_losses, label="training loss", linestyle="solid", color="green")
    plt.plot(val_epochs, val_losses, label="validation loss", linestyle="solid", color="blue")
    plt.xlabel("Loss")
    plt.ylabel("Epoch")
    plt.legend(["Train", "Validation"])
    if not os.path.exists(os.path.join("artefacts", "figs", exp_name)):
        os.makedirs(os.path.join("artefacts", "figs", exp_name))
    plt.savefig(os.path.join("artefacts", "figs", exp_name, f"{model_name} loss curve for {nb_epochs} epochs {amp} {seed}.png"))
    # plt.show()
    plt.close()

    # aliasing plot
    test_sine = sf.read(os.path.join(data_path, "test", amp, "test_sine_input_22050.wav"), dtype='float32')[0]
    test_sine_target = sf.read(os.path.join(data_path, "test", amp, "output_sine_test_max_timestep_705600_anti_aliased_22050.wav"), dtype='float32')[0] # output_sine_test_max_timestep_705600_anti_aliased_22050
    test_sine = test_sine[:sample_rate*10]
    test_sine_target = test_sine_target[:sample_rate*10]
    test_out, test_loss = network.predict(test_sine, y_true=test_sine_target, loss_fn=loss_fn, buffer_size=len(test_sine))
    test_out = test_out.flatten().flatten().detach().cpu().numpy()
    fig, axs = plt.subplots()
    axs.set_title("Magnitude Spectrum")
    mark = 12445
    axs.magnitude_spectrum(test_out, Fs=sample_rate, scale='dB', marker='o', markevery=(mark,mark), fillstyle='none', color='black', label=f'{exp_name}_{model_name}')
    test_ticks = np.arange(-130,0,10)
    axs.set_yticks(test_ticks, minor=True)
    axs.set_xlim([0, int(22050/2)]) 
    axs.set_ylim([-130, 0])
    axs.legend()
    axs.grid(which='major', axis='y')
    axs.grid(which='minor', axis='y', linestyle="--")
    plt.savefig(os.path.join("artefacts", "figs", exp_name, f"mag_spec_{model_name}_{nb_epochs}_{amp}_{sample_rate}_{seed}.png"))
    plt.close()

    print(f"Test loss = {test_loss}.")
    if not os.path.exists(os.path.join("artefacts", "wavs", exp_name)):
        os.makedirs(os.path.join("artefacts", "wavs", exp_name))
    sf.write(os.path.join("artefacts", "wavs", exp_name, f"test_{model_name}_{amp}_{nb_epochs}epochs_{seed}.wav"), test_out, sample_rate)
    duration = time() - start
    print(f"Overall training & processing lasted {duration} seconds -> approx. {duration/nb_epochs}s per epoch.")
    print(f"{model_name} trained on {amp} data, for {nb_epochs} epochs at {sample_rate} Hz & train/val split = {ratio} and seed = {seed}.")
    if not os.path.exists(os.path.join("artefacts", "metrics", exp_name)):
        os.makedirs(os.path.join("artefacts", "metrics", exp_name))
    with open(os.path.join("artefacts", "metrics", exp_name, f"{model_name}_{nb_epochs}epochs_{amp}_{seed}.csv"), 'w') as f:
        write = csv.writer(f)
        f.write(exp_name.capitalize())
        write.writerow(train_losses)
        write.writerow(val_losses)
        f.write(f"Overall training & processing lasted {duration} seconds -> approx. {duration/nb_epochs}s per epoch.")
        f.write(f"{model_name} trained on {amp} data, for {nb_epochs} epochs at {sample_rate} Hz & train/val split = {ratio} and seed = {seed}.")
        f.write(f"\nTest loss = {test_loss}.")
    torch.cuda.empty_cache()

def load_and_predict(model_type: str, model_file:str, sample_rate:int, input_path:str):
    input_signal, input_sr = sf.read(input_path, dtype='float32')
    assert input_sr == sample_rate, "Sample rate of model training data doesn't match the sample rate of the chosen test input."
    torch.cuda.set_device(0)
    assert model_type in ["LSTM", "CustomLSTM", "WaveNet"], "This model doesn't exist."
    if model_type == "LSTM":
        model = models.LSTM(custom = False)
    elif model_type == "CustomLSTM":
        model = models.LSTM(custom = True)
    elif model_type == "WaveNet":
        model = models.WaveNet()
    model = model.cuda()
    chck_pt = torch.load(model_file)
    model.load_state_dict(chck_pt)
    return model.predict(input_signal)

if __name__ == "__main__":
    main("LSTM", 0)
