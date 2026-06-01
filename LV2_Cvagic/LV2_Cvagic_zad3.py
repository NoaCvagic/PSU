import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("tiger.png")

img = img[:, :, 0].copy()

bright = np.clip(img + 50, 0, 255)

plt.figure()
plt.imshow(bright, cmap="gray")
plt.title("Posvijetljena slika")


rotirana = np.rot90(img, -1)

plt.figure()
plt.imshow(rotirana, cmap="gray")
plt.title("Rotirana slika")

zrcaljena = np.fliplr(img)

plt.figure()
plt.imshow(zrcaljena, cmap="gray")
plt.title("Zrcaljena slika")


manja = img[::10, ::10]

plt.figure()
plt.imshow(manja, cmap="gray")
plt.title("Smanjena rezolucija")

nova = np.zeros_like(img)

sirina = img.shape[1]

pocetak = sirina // 4
kraj = sirina // 2

nova[:, pocetak:kraj] = img[:, pocetak:kraj]

plt.figure()
plt.imshow(nova, cmap="gray")
plt.title("Druga cetvrtina slike")

plt.show()
