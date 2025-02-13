import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PhysicsSimulator.SinglePendulum.SinglePendulum import SinglePendulum

import tensorflow as tf
print(tf.__version__)

MASS = 0.6
LENGTH = 2.
DRAG = 0.
model = SinglePendulum(0, 0, mass=MASS, length=LENGTH, drag=DRAG)

# x1: theta, x2: theta_dot

MAX_x1, MIN_x1 = (2 * np.pi, -2 * np.pi)
DELTA_x1 = np.pi/60
MAX_x2, MIN_x2 = (6 * np.pi, -6 * np.pi)
DELTA_x2 = np.pi/20

x1_set = np.arange(MIN_x1, MAX_x1 + DELTA_x1, DELTA_x1)
x2_set = np.arange(MIN_x2, MAX_x2 + DELTA_x2, DELTA_x2)

# u: control input

# u_set = np.array([-2., 0., 2.])
# u_P_list = np.array([1., 1., 1.])
d = 0.1
u_set = np.arange(-2., 2. + d, d)
u_P_list = np.full(u_set.shape, 1.)
u_P_set = u_P_list/u_P_list.sum()


DELTA_t = 0.0025 # for Integration

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
    # x1_n = 1
    # x2_n = 1
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

x1_dot_space_set = -np.array(x1_dot_space_set)
x2_dot_space_set = -np.array(x2_dot_space_set)

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

class Concentration(object):
    def __init__(self, init_concentration, target_point, dot_space_set, u_P_set_list, DELTA_x, DELTA_t):

        self.x1_dot_space_set = tf.constant(dot_space_set[0], dtype=tf.float32)
        self.x2_dot_space_set = tf.constant(dot_space_set[1], dtype=tf.float32)
        self.u_P_set_list = tf.constant(u_P_set_list, dtype=tf.float32)

        is_x1_dot_set_positive = np.full(dot_space_set[0].shape, 1.)
        is_x1_dot_set_negative = np.full(dot_space_set[0].shape, 1.)
        is_x2_dot_set_positive = np.full(dot_space_set[1].shape, 1.)
        is_x2_dot_set_negative = np.full(dot_space_set[1].shape, 1.)

        is_x1_dot_set_positive[np.where(dot_space_set[0] < 0)] = 0
        is_x1_dot_set_negative[np.where(dot_space_set[0] > 0)] = 0
        is_x2_dot_set_positive[np.where(dot_space_set[1] < 0)] = 0
        is_x2_dot_set_negative[np.where(dot_space_set[1] > 0)] = 0

        self.is_x1_dot_set_positive = tf.constant(is_x1_dot_set_positive, dtype=tf.float32)
        self.is_x1_dot_set_negative = tf.constant(is_x1_dot_set_negative, dtype=tf.float32)
        self.is_x2_dot_set_positive = tf.constant(is_x2_dot_set_positive, dtype=tf.float32)
        self.is_x2_dot_set_negative = tf.constant(is_x2_dot_set_negative, dtype=tf.float32)

        boundary_manage = np.full(init_concentration.shape, 1.)
        boundary_manage[0, :] = 0
        boundary_manage[-1, :] = 0
        boundary_manage[:, 0] = 0
        boundary_manage[:, -1] = 0
        self.boundary_manage = tf.constant(boundary_manage, dtype=tf.float32)

        self.target_point = [n[0] for n in target_point]
        self.DELTA_x1 = DELTA_x[0]
        self.DELTA_x2 = DELTA_x[1]
        self.DELTA_t = DELTA_t

        self.loop_count = tf.constant(0)
        self.loop_should_stop = lambda i: tf.less(i, 1000)
        self.loop_body = lambda i: self.update(i)

        self.toropogical_space_concentration = tf.Variable(np.full(init_concentration.shape, 1.0), dtype=tf.float32)
        self.step_div = tf.constant(np.full(init_concentration.shape, 1.0), dtype=tf.float32)
        self.update(1)
        self.step_div = tf.constant(self.toropogical_space_concentration.numpy(), dtype=tf.float32)

        self.toropogical_space_concentration = tf.Variable(init_concentration, dtype=tf.float32)

    @tf.function
    def update_1000(self):
        tf.while_loop(self.loop_should_stop, self.loop_body, [self.loop_count])

    @tf.function
    def update(self, i):
        d_positive_x1_dot_concentration = tf.roll(self.toropogical_space_concentration, 1, axis=0) * tf.math.abs(self.x1_dot_space_set) * self.is_x1_dot_set_positive
        d_negative_x1_dot_concentration = tf.roll(self.toropogical_space_concentration, -1, axis=0) * tf.math.abs(self.x1_dot_space_set) * self.is_x1_dot_set_negative
        d_decrease_concentration = self.toropogical_space_concentration * tf.abs(self.x1_dot_space_set)
        d_toropogical_space_concentration_x1 = (d_positive_x1_dot_concentration + d_negative_x1_dot_concentration - d_decrease_concentration) * self.u_P_set_list
        d_toropogical_space_concentration_x1 = tf.reduce_sum(d_toropogical_space_concentration_x1, 0) * self.DELTA_t/self.DELTA_x1


        # about x2 axis
        # TODO: make same func for every x_n
        d_positive_x2_dot_concentration = tf.roll(self.toropogical_space_concentration, 1, axis=1) * tf.abs(self.x2_dot_space_set) * self.is_x2_dot_set_positive
        d_negative_x2_dot_concentration = tf.roll(self.toropogical_space_concentration, -1, axis=1) * tf.abs(self.x2_dot_space_set) * self.is_x2_dot_set_negative
        d_decrease_concentration = self.toropogical_space_concentration * tf.abs(self.x2_dot_space_set)
        d_toropogical_space_concentration_x2 = (d_positive_x2_dot_concentration + d_negative_x2_dot_concentration - d_decrease_concentration) * self.u_P_set_list
        d_toropogical_space_concentration_x2 = tf.reduce_sum(d_toropogical_space_concentration_x2, 0) * self.DELTA_t/self.DELTA_x2

        d_toropogical_space_concentration = d_toropogical_space_concentration_x1 + d_toropogical_space_concentration_x2
        self.toropogical_space_concentration.assign_add(d_toropogical_space_concentration)
        self.toropogical_space_concentration[self.target_point].assign(1)
        self.toropogical_space_concentration.assign(self.toropogical_space_concentration * self.boundary_manage * self.step_div)
        return i + 1


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

# toropogical_space_concentration2 = np.array([[1.0 if is_target_element(x1, x2) else 0.0 for x2 in x2_set] for x1 in x1_set])
concentration = Concentration(toropogical_space_concentration, target_point, [x1_dot_space_set, x2_dot_space_set], u_P_set_list, [DELTA_x1, DELTA_x2], DELTA_t)
print(concentration.step_div.numpy())
show_plot(concentration.step_div.numpy())
# exit()

for n in tqdm(range(2000)):
    # toropogical_space_concentration2 = uptade_concentration(toropogical_space_concentration2)
    concentration.update_1000()
    # if n % 50 == 0:
    #     toropogical_space_concentration_data = concentration.toropogical_space_concentration.numpy()
    #     show_plot(toropogical_space_concentration_data)
            # print(np.sum(np.abs(toropogical_space_concentration2 - toropogical_space_concentration_data)))
        # show_plot(toropogical_space_concentration2)

toropogical_space_concentration_data = concentration.toropogical_space_concentration.numpy()
show_plot(toropogical_space_concentration_data)

import os
import glob
import json
import datetime

dt_now = datetime.datetime.now()

file_list = glob.glob("./*")

n = 1
while(True):
    path = "./toropogical_space_concentration" + str(n)
    n += 1
    if not path in file_list:
        print(path)
        os.mkdir(path)
        os.chdir(path)
        np.save("concentration", toropogical_space_concentration_data)


        param = {"datetime": str(dt_now),
                 "MAX_x1": MAX_x1, "MIN_x1": MIN_x1, "DELTA_x1": DELTA_x1,
                 "MAX_x2": MAX_x2, "MIN_x2": MIN_x2, "DELTA_x2": DELTA_x2,
                 "MASS": MASS, "LENGTH": LENGTH, "DRAG": DRAG, "u_set": u_set.tolist()}

        with open('param.json', 'w') as json_file:
            json.dump(param, json_file)
        break



