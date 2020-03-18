import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PhysicsSimulator.SinglePendulum.SinglePendulum import SinglePendulum

model = SinglePendulum(-np.pi, 0., mass=0.2, length=2, drag=0.) # TODO: 変数化する
u_set = np.array([-2., 0., 2.])
# d = 0.1
# u_set = np.arange(-2, 2 + d, d)

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

def show_plot_in_range(concentration, max_value):
    plt.figure(figsize=(7,5))
    ex = [MIN_x1, MAX_x1, MIN_x2, MAX_x2]
    plt.imshow(np.flip(np.where(concentration > max_value, max_value, concentration).T, 0),extent=ex,interpolation='nearest',cmap='Blues',aspect=(MAX_x1-MIN_x1)/(MAX_x2-MIN_x2),alpha=1)

    plt.colorbar()
    plt.show()

show_plot(toropogical_space_concentration)
max_value = 0.005
show_plot_in_range(toropogical_space_concentration, max_value)

x1_concentration_grad, x2_concentration_grad = np.gradient(toropogical_space_concentration, DELTA_x1, DELTA_x2)

def show_quiver(x, y):
    fig, ax = plt.subplots()
    x1_n = int(x1_set.size/40)
    x2_n = int(x2_set.size/40)
    fig_x1_set = x1_set[::x1_n]
    fig_x2_set = x2_set[::x2_n]
    fig_velocoty_x1_dot_set = x[::x1_n, ::x2_n].T
    fig_velocoty_x2_dot_set = y[::x1_n, ::x2_n].T

    q = ax.quiver(fig_x1_set, fig_x2_set, fig_velocoty_x1_dot_set, fig_velocoty_x2_dot_set)
    ax.quiverkey(q, X=0.3, Y=1.1, U=10, label='Quiver key, length = 10', labelpos='E')

    plt.show()

show_quiver(x1_concentration_grad, x2_concentration_grad)

size_grad = np.sqrt(x1_concentration_grad ** 2 + x2_concentration_grad ** 2)

x1_concentration_grad = x1_concentration_grad / size_grad
x2_concentration_grad = x2_concentration_grad / size_grad

x1_concentration_grad[np.where(x1_concentration_grad == 0)] = 0
x2_concentration_grad[np.where(x2_concentration_grad == 0)] = 0

show_quiver(x1_concentration_grad, x2_concentration_grad)

def value2list_num(x1, x2):
    x1_num = int((x1 - MIN_x1)/DELTA_x1)
    x2_num = int((x2 - MIN_x2)/DELTA_x2)
    return (x1_num, x2_num)

def input(theta, theta_dot):
    def norm_x(theta, theta_dot, u):
        x = model.singlependulum_dynamics(theta, theta_dot, u)
        return x/np.linalg.norm(x)
    x_set = np.array([norm_x(theta, theta_dot, u) for u in u_set])
    list_num = value2list_num(theta, theta_dot)
    # print(theta, theta_dot)
    desired_direction = np.array([x1_concentration_grad[list_num], x2_concentration_grad[list_num]])
    direction_diff = [np.dot(x, desired_direction) for x in x_set]
    u = u_set[np.where(direction_diff == max(direction_diff))]
    if u.size == 1:
        u = u[0]
    else:
        u = 0
    print(u)
    return u

input_set = np.array([[input(theta, theta_dot) for theta_dot in x2_set] for theta in x1_set])

print(input_set)
show_plot(input_set)

# start simulation
singlePendulum = model

time = 0.
dt = 10**(-2)
max_step = 20 * 10**(2) + 1

df = pd.DataFrame(columns=['time', 'theta', 'theta_dot', 'input'])

# def add_data(df):

for s in range(0, max_step):
    time = s * dt
    singlePendulum.input = input_set[value2list_num(singlePendulum.state[0], singlePendulum.state[1])]
    # singlePendulum.input = 4
    tmp_data = tuple([time]) + singlePendulum.state + tuple([singlePendulum.input])
    print(time)
    tmp_se = pd.Series(tmp_data, index=df.columns)
    df = df.append(tmp_se, ignore_index=True)
    singlePendulum.step(dt)

# # df.to_csv("./data.csv", index=False)
df.plot(x='time', y='theta')
df.plot(x='time', y='theta_dot')
df.plot(x='time', y='input')
df.plot(x='theta', y='theta_dot')
# show_plot_in_range(toropogical_space_concentration, max_value)
# show_plot(input_set)
# plt.show()

fig = plt.figure(figsize=(7,5))
ax = fig.subplots()
ex = [MIN_x1, MAX_x1, MIN_x2, MAX_x2]
ax.imshow(np.flip(input_set.T, 0),extent=ex,interpolation='nearest',cmap='Blues',aspect=(MAX_x1-MIN_x1)/(MAX_x2-MIN_x2),alpha=1)

# ax2 = ax.twinx()
ax.plot(df["theta"], df["theta_dot"], linewidth=1, color="crimson")
# ax2.set_ylim([MIN_x2, MAX_x2])
plt.show()