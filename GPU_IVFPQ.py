import faiss, pdb
import numpy as np
from time import time

# prepare input
d = 1408
n_data = 1000 * 1000 * 2
query = np.random.random((1,d)).astype(np.float32)
data = np.random.random((n_data,d)).astype(np.float32)



# gpu settings
"""
# training part
index = faiss.index_factory(d, "IVF64,PQ64")
# IVF: partitioning the index into clusters and limiting the search to only a few clusters.
# PQ: compress the vectors by partitioning the vector into smaller subvectors, perform k-means clustering, and use the centroids of these clusters to represent the vectors. 
co = faiss.GpuClonerOptions()
co.useFloat16 = True
co.useFloat16LookupTables = True
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, index, co)
gpu_index.train(data)
gpu_index.add(data)
faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), "gpu_index.bin")
"""

# load from checkpoint
index = faiss.read_index("gpu_index.bin")
co = faiss.GpuClonerOptions()
co.useFloat16 = True
co.useFloat16LookupTables = True
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, index, co)

# start
start = time()
dis, ind = gpu_index.search(query, 200)
print("%f secs"%(time()-start))#0.001395 secs
#(IVF,PQ)=(32,32)=>0.001395 secs
#(IVF,PQ)=(32,64)=>0.001167 secs
#(IVF,PQ)=(64,64)=>0.000524 secs
