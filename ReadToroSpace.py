import numpy as np
import matplotlib.pyplot as plt

toropogical_space_concentration = np.load("toropogical_space_concentration.npy")

MAX_x1, MIN_x1 = (2 * np.pi, -2 * np.pi)
DELTA_x1 = np.pi/20
MAX_x2, MIN_x2 = (10 * np.pi, -10 * np.pi)
DELTA_x2 = np.pi/5

x1_set = np.arange(MIN_x1, MAX_x1 + DELTA_x1, DELTA_x1)
x2_set = np.arange(MIN_x2, MAX_x2 + DELTA_x2, DELTA_x2)

def show_plot(concentration):
    plt.figure(figsize=(7,5))
    ex = [MIN_x1, MAX_x1, MIN_x2, MAX_x2]
    plt.imshow(np.flip(concentration.T, 0),extent=ex,interpolation='nearest',cmap='Blues',aspect=(MAX_x1-MIN_x1)/(MAX_x2-MIN_x2),alpha=1)

    plt.colorbar()
    plt.show()

show_plot(toropogical_space_concentration)

x1_concentration_grad, x2_concentration_grad = np.gradient(toropogical_space_concentration, DELTA_x1, DELTA_x2)


fig, ax = plt.subplots()
x1_n = int(x1_set.size/40)
x2_n = int(x2_set.size/40)
fig_x1_set = x1_set[::x1_n]
fig_x2_set = x2_set[::x2_n]
fig_velocoty_x1_dot_set = -x1_concentration_grad[::x1_n, ::x2_n].T
fig_velocoty_x2_dot_set = -x2_concentration_grad[::x1_n, ::x2_n].T

q = ax.quiver(fig_x1_set, fig_x2_set, fig_velocoty_x1_dot_set, fig_velocoty_x2_dot_set)
ax.quiverkey(q, X=0.3, Y=1.1, U=10, label='Quiver key, length = 10', labelpos='E')

plt.show()