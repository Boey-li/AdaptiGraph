import h5py
import pickle
import numpy as np



def store_data_pickle(data_names, data, path):
    d = {}
    for i in range(len(data_names)):
        d[data_names[i]] = data[i]
    pickle.dump(d, open(path, 'wb'))


def store_data(data_names, data, path):
    hf = h5py.File(path, 'w')
    for i in range(len(data_names)):
        hf.create_dataset(data_names[i], data=data[i])
    hf.close()


def load_data(data_names, path):
    hf = h5py.File(path, 'r')
    data = []
    for i in range(len(data_names)):
        d = np.array(hf.get(data_names[i]))
        data.append(d)
    hf.close()
    return data
