import faiss
import numpy as np
from time import time

d = 1408
n_data = 1000 * 1000 * 2
#"""
query = np.random.randint(0, 256, (1000, d // 8 )).astype('uint8')
data = np.random.randint(0, 256, (n_data, d // 8)).astype('uint8')
index = faiss.index_binary_factory(d, "BFlat")
index.add(data)
#faiss.write_index_binary(index,"index.bin")
#index = faiss.read_index_binary("index.bin")
start = time()
dis, ind = index.search(query,k=200)
print("%f secs"%(time()-start))#11 secs
del query
del data
#"""


query = np.random.random((1000,d)).astype(np.float32)
data = np.random.random((n_data,d)).astype(np.float32)
#https://github.com/facebookresearch/faiss/issues/95
#https://github.com/facebookresearch/faiss/issues/61
faiss.normalize_L2(data)#in-place normalization
faiss.normalize_L2(query)

#index = faiss.IndexFlatIP(d)
index = faiss.index_factory(d, "Flat", faiss.METRIC_INNER_PRODUCT)
#res = faiss.StandardGpuResources()
#gpu_index = faiss.index_cpu_to_gpu(res, 0, index)

index.add(data)# add vectors to the index
start = time()
dis, ind = index.search(query,k=200)
print("%f secs"%(time()-start))# 8 secs
del query
del data




