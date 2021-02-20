import faiss, pdb
import numpy as np
from time import time

d = 1408
n_data = 1000 * 1000 * 2
query = np.random.random((1000,d)).astype(np.float32)
"""
data = np.random.random((n_data,d)).astype(np.float32)
index = faiss.index_factory(d, "IVF64,PQ64")
# IVF: partitioning the index into clusters and limiting the search to only a few clusters.
# PQ: compress the vectors by partitioning the vector into smaller subvectors, perform k-means clustering, and use the centroids of these clusters to represent the vectors. 
index.train(data)
index.add(data)
faiss.write_index(index,"index.bin")
"""
index = faiss.read_index("index.bin")

start = time()
dis, ind = index.search(query, 200)
print("%f secs"%(time()-start))#0.301116 secs

pdb.set_trace()
