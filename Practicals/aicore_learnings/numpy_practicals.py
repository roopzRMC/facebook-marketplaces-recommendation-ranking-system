# %%
import numpy as np

my_array = np.array((1,2,3,4,5))
# %%
my_array
# %%
## display the 5th element
my_array[4]
# %%
# Use the np.random.randint function to create a 2D array of size (3,4) with random integers between 0 and 10.

my_random_array = np.random.randint(0, 10, size=(3,4,2))
# %%
my_random_array

# %%
my_random_array.shape
# %%
"""
Create a NumPy array of random numbers with dimensions 6 x 10. Call it my_array
Slice out the 1st, 3rd and 5th column of the array into as new array. Call it new_array
Print the shape of each array you have created.
Try to multiply my_array x new_array. Does it work? If not, do you know why?
Transpose new_array and try again. Does it work now?

"""

my_array = np.random.randint(0, 20, size=(6,10))
# %%
print(my_array)
# %%
array_subset = my_array[:,[0,2,4]]

# %%
array_subset.shape
# %%
tr_array_subset = array_subset.transpose()
# %%
new_array = array_subset @ tr_array_subset
# %%
new_array
# %%
tr_array_subset
# %%
array_subset
# %%
'''
Create a NumPy array of dimensions 5 x 5 where each element is a zero.
Replace the left diagonal of the array with 1s.

'''

my_z_array = np.zeros(shape=(5,5))
# %%
my_z_array
# %%
my_z_array[(0,1,2,3,4),(0,1,2,3,4)] = 1
# %%
my_z_array
# %%

'''
Create a 3D array of size (3,4,5) and use slicing to select the first two elements of the last dimension.

'''

my3d_array = np.random.randint(4, 9, size=(3,4,5))
# %%
my3d_array
# %%

my3d_array[2][0,0:2]
# %%

### reshaping
arr = np.arange(18)
# %%
arr.shape
# %%
arr
# %%
arr.strides
# %%
reshaped = arr.reshape(3, 2, -1)
# %%
reshaped
# %%
reshaped.strides
# %%
arr1d_int = np.array([1, 2, 3, 4])
arr2d_float = np.array(((1, 2, 3, 4), (5, 6, 7, 8.0)))  # Notice 8.0
# %%
arr2d_float
# %%
arr
# %%
arr.reshape(3,2,-1)
# %%
type(arr)
# %%
arr.strides
# %%
arr2 = arr[::2]
arr2
# %%
arr2.strides
# %%
'''
Create a NumPy 1D-array with 10 elements, then reshape the array into a 2x5 matrix.
Create a second NumPy array with 5 elements, and join it to the first array using the concatenate function.
'''

my1d_array = np.random.randint(1, 10, size=(1, 10))
# %%
my1d_array_reshape = my1d_array.reshape(2,5)
# %%
my1d_array_reshape
# %%
my5e_array = np.random.randint(4,8, size=(1,5))
# %%
print(my5e_array)
# %%
new_array = np.concatenate((my1d_array_reshape, my5e_array))
# %%
new_array
# %%
'''
Create two arrays of size (2,3) and broadcast them together to create an array of size (2,3,3).
'''
array_1 = np.random.randint(5, 12, size=(2,3))
array_2 = np.random.randint(2, 8, size=(2,3))

# %%

# %%
ta1 = array_1.reshape(2, 1, 3)
# %%
ta2 = array_2.reshape(2,3,1)
# %%
result = ta1 * ta2
# %%
result.shape
# %%
result
# %%
