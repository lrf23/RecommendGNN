import numpy as np
import scipy.sparse as sp

# 非零元素的行索引
row = np.array([0,0,0,0,2,2])
# 非零元素的列索引
col = np.array([0,1,2,3,4,3])
# 非零元素的值
data = np.array([4, 5, 6, 7,7,7])

# 创建 COO 矩阵
coo = sp.coo_matrix((data, (row, col)), shape=(6, 6))

# print(coo[(0,0)])
# dok_matrix =coo.todok()

# print(dok_matrix[(0,0)])
csr_mat = coo.tocsr()

# 打印 CSR 矩阵
print(csr_mat)
print("CSR matrix data:", csr_mat.data)
print("CSR matrix indices:", csr_mat.indices)
print("CSR matrix indptr:", csr_mat.indptr)

csrmat = (coo.tocsr() != 0) * 1.0
print(np.reshape(csrmat[0].toarray(), [-1]))