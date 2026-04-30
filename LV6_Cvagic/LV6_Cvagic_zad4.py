import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans

imageNew = mpimg.imread('example_grayscale.png')

X = imageNew.reshape(-1, 1)

n_boje = 10
kmeans = KMeans(n_clusters=n_boje, n_init=10, random_state=42)
kmeans.fit(X)

image_compressed = kmeans.cluster_centers_[kmeans.labels_]
image_compressed = image_compressed.reshape(imageNew.shape)

plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(imageNew, cmap='gray')

plt.subplot(1, 2, 2)
plt.title(f"Kopija ({n_boje} boja)")
plt.imshow(image_compressed, cmap='gray')
plt.show()