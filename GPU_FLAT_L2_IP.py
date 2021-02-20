import faiss, torch
import numpy as np
from time import time
# prepare input
d = 1408
n_data = 1000 * 1000 * 2
query = np.random.random((1,d)).astype(np.float32)
data = np.random.random((n_data,d)).astype(np.float32)


# L2
"""
# gpu setting
flat_config = faiss.GpuIndexFlatConfig()
flat_config.useFloat16=True
res = faiss.StandardGpuResources()
gpu_index = faiss.GpuIndexFlatL2(res, d, flat_config)
gpu_index.add(data)# add vectors to the index

# start
start = time()
dis, ind = gpu_index.search(query,k=200)
print("%f secs"%(time()-start))#0.015792 secs
"""

# Inner Product
# preprocess
faiss.normalize_L2(data)
faiss.normalize_L2(query)
# gpu setting
flat_config = faiss.GpuIndexFlatConfig()
flat_config.useFloat16=True
res = faiss.StandardGpuResources()
gpu_index = faiss.GpuIndexFlatIP(res, d, flat_config)
gpu_index.add(data)# add vectors to the index
# start
start = time()
dis, ind = gpu_index.search(query,k=200)
print("%f secs"%(time()-start))#0.015208 sec


