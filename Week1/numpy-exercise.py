import numpy as np

##Cau 1: Thay the tat ca cac phan tu trong
#mang co gia tri > 30 thanh 30, <10 thanh 10
a1 = np.array([28, 15, 6, 7, 10, 14, 56, 75, 2, 16, 18])

a1 = np.where(a1 >= 10, a1, 10)
a1 = np.where(a1 <= 30, a1, 30)

print(a1)

##Cau 2: Lay vi tri cua 3 phan tu lon nhat trong mot mang numpy
a2 = np.array([28, 15, 6, 7, 10, 14, 56, 75, 2, 16, 18])

#np.argpatition(arr): mang index theo thu tu tang dan
out = np.argpartition(a2, -3)[-3:]

print(out)

##Cau 3: Chuyen mang nhieu chieu thanh mang 1 vecto
a3 = np.array([[28, 15, 6],
               [7, 10, 14],
               [56, 75, 2]])

a3 = a3.flatten()
print(a3)

##Cau 4: