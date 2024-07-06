# ------------------------------------------------------------------------------
# Copyright
#
# This file is part of the repository of the CVPR'24 paper:
# "Multiple View Geometry Transformers for 3D Human Pose Estimation"
# https://github.com/XunshanMan/MVGFormer
#
# Please follow the LICENSE detail in the main repository.
# ------------------------------------------------------------------------------

import torch
import time

# try 1: 
# import sys
# sys.path.append('/X/run/torch-batch-svd')
# from torch_batch_svd import svd

for device in ['cpu', 'cuda']:
    print('device=', device)

    # queries, joints, matrix(10x4)
    data_mat = torch.rand(900,15,10,4).to(device)

    t1 = time.time()
    data_mat_flat = data_mat.flatten(0,1)
    torch.svd(data_mat_flat)
    t2 = time.time()
    print('torch.svd batch:', t2-t1)

    # for loop
    t1 = time.time()
    for i in range(data_mat.size(0)):
        data_cur = data_mat[i,...]
        torch.svd(data_cur)
    t2 = time.time()
    print('torch.svd forloop:', t2-t1)

    t1 = time.time()
    data_mat_flat = data_mat.flatten(0,1)
    torch.linalg.svd(data_mat_flat)
    t2 = time.time()
    print('torch.linag.svd batch:', t2-t1)

    # for loop
    t1 = time.time()
    for i in range(data_mat.size(0)):
        data_cur = data_mat[i,...]
        torch.linalg.svd(data_cur)
    t2 = time.time()
    print('torch.linag.svd forloop:', t2-t1)
