import numpy as np

a = [2,1,1,2,3]
b = np.array([4,5])
#a = np.concatenate([a,b])
#print(a)
del a[1]
print(a)

x = np.array([1,2,3]*5)
y = np.array([1]*15)
x += y
print(x)