# Pengolahancitra14

NAMA  : DIMAS ADITYA PUTRANTO

KELAS : TI.22.A5

NIM   : 312210489

# Tugas
Masukan source code berikut. 

```
import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assign and open image from URL
url = 'https://i.pinimg.com/564x/73/6c/b8/736cb88a471f0ac9354ace1886ebab18.jpg'
response = requests.get(url, stream=True)

# Saving the image
with open('image.png', 'wb') as f:
    f.write(response.content)

# Read the image using OpenCV
img = cv2.imread('image.png')

# Converting the image into gray scale for faster computation.
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Calculating the SVD
u, s, v = np.linalg.svd(gray_image, full_matrices=False)

# Inspect shapes of the matrices
print(f'u.shape:{u.shape}, s.shape:{s.shape}, v.shape:{v.shape}')

# Calculate variance explained by each singular value
var_explained = np.round(s*2 / np.sum(s*2), decimals=6)

# Variance explained top Singular vectors
print(f'Variance Explained by Top 20 singular values:\n{var_explained[0:20]}')

# Plotting the variance explained by the top 20 singular values
sns.barplot(x=list(range(1, 21)), y=var_explained[0:20], color="dodgerblue")

plt.title('Variance Explained Graph')
plt.xlabel('Singular Vector', fontsize=16)
plt.ylabel('Variance Explained', fontsize=16)
plt.tight_layout()
plt.show()

# Plot images with different number of components
comps = [3648, 1, 5, 10, 15, 20]
plt.figure(figsize=(12, 6))

for i in range(len(comps)):
    low_rank = u[:, :comps[i]] @ np.diag(s[:comps[i]]) @ v[:comps[i], :]

    plt.subplot(2, 3, i + 1)
    plt.imshow(low_rank, cmap='gray')
    if i == 0:
        plt.title(f'Actual Image with n_components = {comps[i]}')
    else:
        plt.title(f'n_components = {comps[i]}')

plt.tight_layout()
plt.show()
```

# Hasil
Maka akan menjadi seperti ini

- Variance Explained Graph

![1000074578](https://github.com/DimasAditya04/Pengolahancitra14/assets/130146099/69b92a45-58f6-4b2f-b3a3-a46c1708cbe9)

- Singular Vector
![1000074579](https://github.com/DimasAditya04/Pengolahancitra14/assets/130146099/29c28197-6874-4f67-abb4-041108e1da34)
