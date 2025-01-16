import numpy as np

data_load = np.load('MLTB_data.npz', allow_pickle=True)
data_dict = data_load['Dataset'].item()
dataset = list(data_dict.values())
print(dataset)

