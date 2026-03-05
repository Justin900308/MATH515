import numpy as np
import scipy.sparse as sp

def load(filename):
    """
    Loads the total number of measurements, user IDs, movie IDs, 
    and corresponding ratings from the binary file.
    """
    with open(filename, "rb") as fid:
        # Read the total number of measurements (1 Int64 value)
        l = np.fromfile(fid, dtype=np.int64, count=1)[0]
        
        # Read the arrays based on the total length
        uid = np.fromfile(fid, dtype=np.int64, count=l)
        mid = np.fromfile(fid, dtype=np.int64, count=l)
        rat = np.fromfile(fid, dtype=np.int64, count=l)
        
    return l, uid, mid, rat

def partition(l):
    """
    Partitions the data indices into an 80/20 train/test split.
    Note: Python is 0-indexed, so the slicing is slightly adjusted from Julia.
    """
    np.random.seed(123)
    chaos = np.random.permutation(l)
    i = int(np.floor(0.8 * l))
    
    ind_tr = chaos[:i]
    ind_te = chaos[i:]
    
    return ind_tr, ind_te

def spmatGen(ind, uid, mid, rat):
    """
    Generates a sparse matrix from the given indices.
    """
    uidP = uid[ind]
    midP = mid[ind]
    ratP = rat[ind]
    
    # Generate the sparse matrix in COO format, then convert to CSR for efficiency
    return sp.coo_matrix((ratP, (uidP, midP))).tocsr()