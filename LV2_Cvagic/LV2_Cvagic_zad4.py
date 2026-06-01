import numpy as np
import matplotlib.pyplot as plt

def sahovnica(velicina, redovi, stupci):

    crni = np.zeros((velicina, velicina))
    bijeli = np.ones((velicina, velicina)) * 255

    retci = []

    for i in range(redovi):

        elementi = []

        for j in range(stupci):

            if (i + j) % 2 == 0:
                elementi.append(crni)
            else:
                elementi.append(bijeli)

        retci.append(np.hstack(elementi))

    img = np.vstack(retci)

    return img


img = sahovnica(50, 4, 5)

plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.show()