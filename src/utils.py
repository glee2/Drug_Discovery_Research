import h5py
from scipy.sparse import csr_matrix, lil_matrix

def save_hdf5(data_to_save, data_path):
    with h5py.File(data_path, "w") as f:
        if isinstance(data_to_save, (csr_matrix, lil_matrix)):
            f.create_dataset("data", data=data_to_save.data)
            f.create_dataset("indices", data=data_to_save.indices)
            f.create_dataset("indptr", data=data_to_save.indptr)
            f.attrs["shape"] = data_to_save.shape
            f.attrs["type"] = "sparse"
        else:
            f.create_dataset("data", data=data_to_save)
            f.attrs["type"] = "dense"
    
def load_hdf5(data_path):
    with h5py.File(data_path, "r") as f:
        if f.attrs["type"]=="sparse":
            data = f["data"][:]
            indices = f["indices"][:]
            indptr = f["indptr"][:]
            shape = f.attrs["shape"]
            return csr_matrix((data, indices, indptr), shape=shape)
        else:
            return f["data"][:]