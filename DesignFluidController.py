# import numpy as np
import cupy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PhysicsSimulator.SinglePendulum.SinglePendulum import SinglePendulum

model = SinglePendulum(0, 0, mass=0.6, length=2, drag=0.)

# x1: theta, x2: theta_dot

MAX_x1, MIN_x1 = (2 * np.pi, -2 * np.pi)
DELTA_x1 = np.pi/20
MAX_x2, MIN_x2 = (10 * np.pi, -10 * np.pi)
DELTA_x2 = np.pi/5

x1_set = np.arange(MIN_x1, MAX_x1 + DELTA_x1, DELTA_x1)
x2_set = np.arange(MIN_x2, MAX_x2 + DELTA_x2, DELTA_x2)

# u: control input

u_set = np.ndarray([-2., 0., 2.])
u_P_list = np.ndarray([1., 1., 1.])
u_P_set = u_P_list/u_P_list.sum()


DELTA_t = 0.004 # for Integration

toropogical_space_velocity = [[[model.singlependulum_dynamics(theta, theta_dot, u) for theta_dot in x2_set] for theta in x1_set] for u in u_set]
toropogical_space_velocity = np.ndarray(toropogical_space_velocity)
print(toropogical_space_velocity)
print(toropogical_space_velocity.shape)

for space_velocity in toropogical_space_velocity:
    velocoty_x1_dot_set = space_velocity[:, :, 0]
    velocoty_x2_dot_set = space_velocity[:, :, 1]
    
    print(x1_set.shape)
    print(x2_set.shape)
    print(velocoty_x1_dot_set.shape)
    print(velocoty_x2_dot_set.shape)

    fig, ax = plt.subplots()
    x1_n = int(x1_set.size/10)
    x2_n = int(x2_set.size/10)
    fig_x1_set = x1_set[::x1_n]
    fig_x2_set = x2_set[::x2_n]
    fig_velocoty_x1_dot_set = velocoty_x1_dot_set[::x1_n, ::x2_n].T
    fig_velocoty_x2_dot_set = velocoty_x2_dot_set[::x1_n, ::x2_n].T

    q = ax.quiver(fig_x1_set, fig_x2_set, fig_velocoty_x1_dot_set, fig_velocoty_x2_dot_set)
    ax.quiverkey(q, X=0.3, Y=1.1, U=10, label='Quiver key, length = 10', labelpos='E')

    plt.show()

def is_nera(val_x1, val_x2, wid):
    return abs(val_x1 - val_x2) <= wid/2

def is_target_element(val_x1, val_x2):
    target_x1 = 0.
    target_x2 = 0.
    return is_nera(val_x1, target_x1, DELTA_x1) and is_nera(val_x2, target_x2, DELTA_x2)

toropogical_space_concentration = np.ndarray([[1.0 if is_target_element(x1, x2) else 0.0 for x2 in x2_set] for x1 in x1_set])
target_point = np.where(toropogical_space_concentration == 1)

print(toropogical_space_concentration)
print("toropogical_space_concentration", toropogical_space_concentration.shape)
print("taeget point {}".format(target_point))


def show_plot(concentration):
    plt.figure(figsize=(7,5))
    ex = [MIN_x1, MAX_x1, MIN_x2, MAX_x2]
    plt.imshow(np.flip(concentration.T, 0),extent=ex,interpolation='nearest',cmap='Blues',aspect=(MAX_x1-MIN_x1)/(MAX_x2-MIN_x2),alpha=1)

    plt.colorbar()
    plt.show()

# show_plot(toropogical_space_concentration)
# exit()

x1_dot_space_set, x2_dot_space_set = [], []

for space_velocity in toropogical_space_velocity:
    x1_dot_space_set.append(space_velocity[:, :, 0])
    x2_dot_space_set.append(space_velocity[:, :, 1])

x1_dot_space_set = -np.ndarray(x1_dot_space_set)
x2_dot_space_set = -np.ndarray(x2_dot_space_set)

max_velocity = max(np.max(np.abs(x1_dot_space_set)), np.max(np.abs(x2_dot_space_set)))
print("param {}".format(max_velocity * DELTA_t/min(DELTA_x1, DELTA_x2)))

print("x1_dot_space_set")
print(x1_dot_space_set.shape)

u_P_set_list = np.zeros((u_P_set.shape + toropogical_space_concentration.shape), dtype=float)
n = 0
for u_P in u_P_set:
    u_P_set_list[n] = u_P
    n += 1

# print("u_P_set_list")
# print(u_P_set_list.shape)
# exit()

def uptade_concentration(concentration_set):

    # about x1 axis

    d_positive_x1_dot_concentration = np.roll(concentration_set, 1, axis=0) * np.abs(x1_dot_space_set)
    d_negative_x1_dot_concentration = np.roll(concentration_set, -1, axis=0) * np.abs(x1_dot_space_set)
    d_decrease_concentration = concentration_set * np.abs(x1_dot_space_set)

    d_positive_x1_dot_concentration[np.where(x1_dot_space_set < 0)] = 0
    d_negative_x1_dot_concentration[np.where(x1_dot_space_set > 0)] = 0

    d_toropogical_space_concentration_x1 = (d_positive_x1_dot_concentration + d_negative_x1_dot_concentration - d_decrease_concentration) * u_P_set_list
    d_toropogical_space_concentration_x1 = np.sum(d_toropogical_space_concentration_x1, axis=0) * DELTA_t/DELTA_x1

    # about x2 axis
    # TODO: make same func for every x_n

    d_positive_x2_dot_concentration = np.roll(concentration_set, 1, axis=1) * np.abs(x2_dot_space_set)
    d_negative_x2_dot_concentration = np.roll(concentration_set, -1, axis=1) * np.abs(x2_dot_space_set)
    d_decrease_concentration = concentration_set * np.abs(x2_dot_space_set)

    d_positive_x2_dot_concentration[np.where(x2_dot_space_set < 0)] = 0
    d_negative_x2_dot_concentration[np.where(x2_dot_space_set > 0)] = 0

    d_toropogical_space_concentration_x2 = (d_positive_x2_dot_concentration + d_negative_x2_dot_concentration - d_decrease_concentration) * u_P_set_list
    d_toropogical_space_concentration_x2 = np.sum(d_toropogical_space_concentration_x2, axis=0) * DELTA_t/DELTA_x2

    concentration_set += d_toropogical_space_concentration_x1 + d_toropogical_space_concentration_x2

    concentration_set[0, :] = 0
    concentration_set[-1, :] = 0
    concentration_set[:, 0] = 0
    concentration_set[:, -1] = 0
    concentration_set[target_point] = 1

    return concentration_set

for n in tqdm(range(20000000)):
    toropogical_space_concentration = uptade_concentration(toropogical_space_concentration)
    # if n % 50000 == 0:
    #     # print(toropogical_space_concentration)
    #     # print("toropogical_space_concentration", toropogical_space_concentration.shape)
    #     show_plot(toropogical_space_concentration)

show_plot(toropogical_space_concentration)

np.save("toropogical_space_concentration2", toropogical_space_concentration)
