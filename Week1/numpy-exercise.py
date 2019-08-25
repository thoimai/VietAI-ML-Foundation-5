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

##Cau 4: Tao  mot one-hot encoding cho mot mang numpy. chuyen doi moi gia tri
# so nguyen n trong mot vector v ma vi tri thu n trong vector v mang gia tri 1, con laij = 0
input_vec = np.array([2,1,3,3,1,2])
output_index = input_vec - 1

size = np.max(input_vec)

output = np.zeros((input_vec.shape[0], size))
output[np.arange(input_vec.shape[0]), output_index] = 1

# for i in range(input_vec.shape[0]):
#     output[i, output_index[i]] = 1

print(output)

##Cau 5: sap xep cac phan tu trong mag
input = [3, 6, 8, 4]
print(np.sort(input))

##Cau 6 : Tim gia tri lon nhat trong mou hang va cot cua mang 2D

input = np.array([[5,1,7],
                  [3,5,2],
                  [6,4,5]])
max_col = np.max(input, axis=0)
max_row = np.max(input, axis=1)
print(max_col, max_row)

##Cau 7: Tim cac phan tu trung lap trong mang
input = np.array([0, 3, 3, 1, 2, 0, 2, 0, 5])

value = np.unique(input)#Lay cac gia tri trong input

output = np.ones(input.shape[0], dtype=bool)

for i in range(input.shape[0]):
    if input[i] in value:
        k = np.argwhere(value == input[i])
        output[i] = False
        value = np.delete(value, k)
        # print(value)

print(output)

##Cau 8: Tru theo dong mang 2 chieu arr2d bang mang 1 chieu arr1d
arr_2D = np.array([[3,3,3],
                  [4,4,4],
                  [5,5,5]])
arr_1D = np.array([1,1,1])
print(arr_2D - arr_1D)

##Cau 9: Bo tat ca gia tri nan tu mot mang numpy
from numpy import nan
input = np.array([1,2,3,nan,5,6,7,nan],dtype=float)
output = input[np.isfinite(input)]
print(output)

##Cau 10: Lay tata ca vi tri noi cac phan tu co gia tri khac nhau
arr1 = np.array([3,4,5,6,7,8])
arr2 = np.array([3,3,5,6,7,9])

same_index = np.where(arr1 != arr2)[0]
print(same_index)