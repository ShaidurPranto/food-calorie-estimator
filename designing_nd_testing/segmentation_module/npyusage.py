import numpy as np
import matplotlib.pyplot as plt

mask = np.load("input_2_side_mask_001.npy") #relative path
# mask = np.load("/home/soham/Downloads/input_2_side_mask_001.npy") #absolute path

print(mask.shape)   # (height, width)
print(mask.dtype)   # bool বা uint8
print(mask.sum())   # কতটা pixel এই segment এ আছে

#print array values
print(mask)

plt.imshow(mask, cmap='gray')
plt.show()