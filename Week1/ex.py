import numpy as np

def get_n_max_element(a, n):
    b=[]
    #np.argpartition(arr, k): copy mang sao cho vi tri thu k giu nguyen vi tri trong mang da sap xep
    #cac phan tu nho hon xep truoc k va lon hon phia sau k mang np.argpatition la mang index so voi mang ban dau
    l = len(a)
    if (l<n):
        return []
    if (l==n):
        return a

    a = np.asarray(a)

    temp = a[np.argpartition(a, l - n)]

    for i in range(l - n, l):
        b.append(temp[i])

    return b

a = [-7, -3, -2, 11, -12, -1, -3, 9, 4, -15, -3, 9, 13, -5, -9, -5, 7, -3, -10, 15]
n = 2
# a = [1, 2, 5, 4, 10, 13, 6, 7, 3]
# n = 4
b = get_n_max_element(a, n)
print(b)

v = np.array([1, 2])
matrix = np.array([[1], [2], [3]])
res = matrix + v
print(res)
print(v.shape[0])
print(matrix.shape[0])
