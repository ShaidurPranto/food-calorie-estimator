import numpy as np
import matplotlib.pyplot as plt

mask = np.load("image_001_mask_000.npy")

print(mask.shape)   # (height, width)
print(mask.dtype)   # bool বা uint8
print(mask.sum())   # কতটা pixel এই segment এ আছে

plt.imshow(mask, cmap='gray')
plt.show()