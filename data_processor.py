import torch
import numpy as np

""" Data Processing """

def reshape_data(input_data, buffer_size = None):
    input_data = np.expand_dims(input_data, 1)
    seg_num = int(np.floor(input_data.shape[0] / buffer_size)) if buffer_size else 1
    if buffer_size:
        output_data = torch.empty((buffer_size, seg_num, 1))
        for i in range(seg_num):
            output_data[:, i, :] = torch.from_numpy(input_data[i*buffer_size: (i+1)*buffer_size, :])
    else: 
        input_data = np.expand_dims(input_data, 1)
        output_data = torch.empty_like(torch.from_numpy(input_data))
        output_data = torch.from_numpy(input_data)
    return output_data
    