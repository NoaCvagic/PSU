import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(open("mtcars.csv", "rb"), 
                  usecols=(1,2,3,4,5,6),
                  delimiter=",",
                  skiprows=1)
wt = data[:,5] * 10
 

plt.scatter(data[:,0], data[:,3], c='g', marker='o', s = wt*10, alpha = 0.5,label='mpg vs hp')

print(data[:,0].min())

print(data[:,0].max())

print(data[:,0].mean())
plt.ylabel('hp')
plt.title('Ovisnost mpg o hp')

cyl = data[:,1]

mpg_6cyl = data[:,0][cyl == 6]

mpg_min_6 = np.min(mpg_6cyl)
mpg_max_6 = np.max(mpg_6cyl)
mpg_mean_6 = np.mean(mpg_6cyl)

print("6 cilindara:")
print("Min mpg:", mpg_min_6)
print("Max mpg:", mpg_max_6)
print("Srednji mpg:", mpg_mean_6)


plt.legend()
plt.show()