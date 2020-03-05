import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
import numpy as np

x = np.arange(3)
print(np.full((2,2), 2, dtype=float))
exit()

x = np.arange(30)
x = x.reshape(3,2,5)

y = np.arange(10)
y = y.reshape(2,5)

z = np.arange(3)

print("x {}, y {}".format(x, y))
print("x * y {}".format(y * x))
print("sum {}".format(np.sum(x*y, axis=0)))
print("x * z {}".format(np.prod(z, x, axis=0)))
exit()

x = np.arange(20)
print(np.roll(x, 2))
print(np.roll(x, -2))

x = x.reshape(4,5)

print(np.roll(x, 1, axis=0))
print(np.roll(x, 1, axis=1))
exit()

xedges = np.array([7, 1, 7, 4, 5, 6, 7])
print(xedges[1: -1])
exit()

xedges = np.array([7, 1])
yedges = np.array([3, -4])
xedges[np.where(yedges < 0)] = 0
print(xedges)
exit()

xedges = np.array([7, 1])
yedges = np.array([[3, -4], [4, 6]])
print(np.dot(xedges, yedges))
exit()

x = np.random.normal(2, 1, 100)
y = np.random.normal(1, 1, 100)
H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
H = H.T  # Let each row list bins with common y range.

print(H, xedges, yedges)
print(x, y)

fig = plt.figure(figsize=(7, 3))
ax = fig.add_subplot(131, title='imshow: square bins')
plt.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

ax = fig.add_subplot(132, title='pcolormesh: actual edges', aspect='equal')
X, Y = np.meshgrid(xedges, yedges)
ax.pcolormesh(X, Y, H)

ax = fig.add_subplot(133, title='NonUniformImage: interpolated', aspect='equal', xlim=xedges[[0, -1]], ylim=yedges[[0, -1]])
im = NonUniformImage(ax, interpolation='bilinear')
xcenters = (xedges[:-1] + xedges[1:]) / 2
ycenters = (yedges[:-1] + yedges[1:]) / 2
im.set_data(xcenters, ycenters, H)
ax.images.append(im)
plt.show()


