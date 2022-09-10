import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker, parafac, partial_tucker
from tensorly.tenalg import mode_dot, multi_mode_dot




def get_tucker_tensors(data_tensor, data_tensor_test, rank=-1, n_iter_max=1000, verbose=2):
    t = data_tensor
    tt = data_tensor_test
    
    if rank is None or rank == -1:
        rank = data_tensor.shape[1:]
        
    if verbose > 1:
        print(f"Tucker decomposition with rank: {rank}. tensor shape: {data_tensor.shape}")
        
    modes = [i for i in range(len(data_tensor.shape))]
    modes = modes[1:]
    core_tucker, factors_tucker = partial_tucker(t, modes, rank=rank, n_iter_max=n_iter_max)
    
    if verbose > 0:
        t_rec_tucker = tl.tucker_to_tensor((core_tucker, [None] + factors_tucker), skip_factor=0)
        print(f"Tucker: Rank: {rank }, rel_error: {tl.norm(t - t_rec_tucker)/tl.norm(t): .5f} ; norm origin: {tl.norm(t)} ; norm recovered: {tl.norm(t_rec_tucker)}")
            
    factors = factors_tucker
    factors = [None] + factors
    tensor_tucker = multi_mode_dot(t, [matrix.T for matrix in factors[1:]], modes=modes)
    tensor_tucker_test = multi_mode_dot(tt, [matrix.T for matrix in factors[1:]], modes=modes)
        
    return tensor_tucker, tensor_tucker_test

def recover_svd(u_s_v, rank=None):
    u, s, v = u_s_v
    if rank is None:
        rank = len(s)
    if len(s) < rank:
        print(f"Worning! Rank [{rank}] is more then len(sigmas) [{len(s)}]!")
        rank = len(s)
        
    recover_mat = np.matmul(u[:, :rank], np.diag(s[:rank]))
    recover_mat = np.matmul(recover_mat, v[:rank, :])
    return recover_mat

def get_SVD_tensors(data_tensor, data_tensor_test, rank=-1,  verbose=2):
    t = data_tensor
    tt = data_tensor_test

    t_uf0 = tl.unfold(t, mode=0)
    tt_uf0 = tl.unfold(tt, mode=0)
    
    if rank is None or rank == -1:
        rank = np.min(t_uf0.shape)
        
    if verbose > 1:
        print(f"SVD decomposition with rank: {rank}. tensor shape: {data_tensor.shape}")
        
    u_0, s_0, v_0 = tl.partial_svd(t_uf0, )

    if verbose > 0:
        t_rec_SVD = recover_svd((u_0, s_0, v_0), rank=rank)
        print(f"SVD: Rank: {rank }, rel_error: {tl.norm(t_uf0 - t_rec_SVD)/tl.norm(t): .5f} ; norm origin: {tl.norm(t_uf0)} ; norm recovered: {tl.norm(t_rec_SVD)}")
            
        matrix_SVD = np.matmul(t_uf0, v_0[:rank, :].T)
        matrix_SVD_test = np.matmul(tt_uf0, v_0[:rank, :].T)
    return matrix_SVD, matrix_SVD_test