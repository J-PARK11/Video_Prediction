from .data_mmnist import load_mmnist
from .data_kth import load_kth
# from .data_taxibj import load_taxibj

def load_data(root, dataname, freq=20, strides=1, current=0, height=120, width=160):
    if dataname == 'mmnist':
        return load_mmnist(root)
    elif dataname == 'kth':
        return load_kth(root, freq, strides, current, height, width)
    else:
        print("No Matching Dataset")
        return

