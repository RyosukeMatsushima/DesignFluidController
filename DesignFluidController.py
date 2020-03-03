import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PhysicsSimulator.SinglePendulum.SinglePendulum import SinglePendulum

model = SinglePendulum(0, 0, mass=0.6, length=2, drag=0.)

# x1: theta, x2: theta_dot

MAX_x1, MIN_x1 = (2 * np.pi, -2 * np.pi)
DELTA_x1 = np.pi/5
MAX_x2, MIN_x2 = (10 * np.pi, -10 * np.pi)
DELTA_x2 = np.pi/5

x1_set = np.arange(MIN_x1, MAX_x1 + DELTA_x1, DELTA_x1)
x2_set = np.arange(MIN_x2, MAX_x2 + DELTA_x2, DELTA_x2)

# u: control input

u_set = np.array([-1., 0., 1.])
u_P_list = np.array([1., 1., 1.])
u_P_set = u_P_list/u_P_list.sum()


DELTA_t = 0.01 # for Integration

toropogical_space_velocity = np.array([[[model.singlependulum_dynamics(theta, theta_dot, u) for theta_dot in x2_set] for theta in x1_set] for u in u_set])
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
    target_x1 = 1.
    target_x2 = 1.
    return is_nera(val_x1, target_x1, DELTA_x1) and is_nera(val_x2, target_x2, DELTA_x2)

toropogical_space_concentration = np.array([[1.0 if is_target_element(x1, x2) else 0.0 for x2 in x2_set] for x1 in x1_set])
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

x1_dot_space_set = np.array(x1_dot_space_set)
x2_dot_space_set = np.array(x2_dot_space_set)

max_velocity = max(np.max(np.abs(x1_dot_space_set)), np.max(np.abs(x2_dot_space_set)))
print("param {}".format(max_velocity * DELTA_t/min(DELTA_x1, DELTA_x2)))

print("x1_dot_space_set")
print(x1_dot_space_set.shape)

def uptade_concentration():
    global toropogical_space_concentration 

    d_toropogical_space_concentration_x1 = np.zeros(toropogical_space_concentration.shape)

    n = 0
    for x1_dot_space in x1_dot_space_set:
        d_positive_x1_dot_concentration = np.roll(toropogical_space_concentration, -1, axis=0) * np.abs(x1_dot_space)
        d_negative_x1_dot_concentration = np.roll(toropogical_space_concentration, 1, axis=0) * np.abs(x1_dot_space)
        d_decrease_concentration = toropogical_space_concentration * np.abs(x1_dot_space)

        d_positive_x1_dot_concentration[np.where(x1_dot_space < 0)] = 0
        d_negative_x1_dot_concentration[np.where(x1_dot_space > 0)] = 0

        # print("x1")
        # print(u_set[n])
        # print("negative value")
        # print(np.where(d_positive_x1_dot_concentration < 0))
        # print(np.where(d_negative_x1_dot_concentration < 0))
        # # print(d_positive_x1_dot_concentration)
        # # print(d_negative_x1_dot_concentration)
        # d_positive_x1_dot_concentration[target_point] = 0.4
        # show_plot(d_positive_x1_dot_concentration)
        # print("s")
        # d_negative_x1_dot_concentration[target_point] = 0.4
        # show_plot(d_negative_x1_dot_concentration)

        d_toropogical_space_concentration_x1 += (d_positive_x1_dot_concentration + d_negative_x1_dot_concentration - d_decrease_concentration) * DELTA_t/DELTA_x1 * u_P_set[n]
        n += 1

    d_toropogical_space_concentration_x2 = np.zeros(toropogical_space_concentration.shape)

    n = 0
    for x2_dot_space in x2_dot_space_set:
        d_positive_x2_dot_concentration = np.roll(toropogical_space_concentration, -1, axis=1) * np.abs(x2_dot_space)
        d_negative_x2_dot_concentration = np.roll(toropogical_space_concentration, 1, axis=1) * np.abs(x2_dot_space)
        d_decrease_concentration = toropogical_space_concentration * np.abs(x2_dot_space)

        d_positive_x2_dot_concentration[np.where(x2_dot_space < 0)] = 0
        d_negative_x2_dot_concentration[np.where(x2_dot_space > 0)] = 0

        # print("x2")
        # print(u_set[n])
        # print("negative value")
        # print(np.where(d_positive_x2_dot_concentration < 0))
        # print(np.where(d_negative_x2_dot_concentration < 0))
        # # print(d_positive_x2_dot_concentration)
        # # print(d_negative_x2_dot_concentration)
        # show_plot(d_positive_x2_dot_concentration)
        # print("s")
        # show_plot(d_negative_x2_dot_concentration)

        d_toropogical_space_concentration_x2 += (d_positive_x2_dot_concentration + d_negative_x2_dot_concentration - d_decrease_concentration) * DELTA_t/DELTA_x2 * u_P_set[n]
        n += 1

    toropogical_space_concentration += d_toropogical_space_concentration_x1 + d_toropogical_space_concentration_x2

    toropogical_space_concentration[0, :] = 0
    toropogical_space_concentration[-1, :] = 0
    toropogical_space_concentration[:, 0] = 0
    toropogical_space_concentration[:, -1] = 0
    toropogical_space_concentration[target_point] = 1


# uptade_concentration()
# exit()

for n in tqdm(range(100000)):
    uptade_concentration()
    if n % 5000 == 0:
        # print(toropogical_space_concentration)
        # print("toropogical_space_concentration", toropogical_space_concentration.shape)
        show_plot(toropogical_space_concentration)

show_plot(toropogical_space_concentration)
