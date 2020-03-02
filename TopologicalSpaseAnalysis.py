import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PhysicsSimulator.SinglePendulum.SinglePendulum import SinglePendulum

model = SinglePendulum(0, 0, mass=0.6, length=1, drag=0.)

max_theta, min_theta = (2 * np.pi, -2 * np.pi)
step_theta = np.pi/10
max_theta_dot, min_theta_dot = (4 * np.pi, -4 * np.pi)
step_theta_dot = np.pi/10

theta_set = np.arange(min_theta, max_theta + step_theta, step_theta)
theta_dot_set = np.arange(min_theta_dot, max_theta_dot + step_theta_dot, step_theta_dot)

toropogical_space_velocity = np.array([[model.singlependulum_dynamics(theta, theta_dot, 0) for theta_dot in theta_dot_set] for theta in theta_set])

velocoty_theta_dot_set = toropogical_space_velocity[:, :, 0]
velocoty_theta_2dot_set = toropogical_space_velocity[:, :, 1]

print(toropogical_space_velocity)

print(theta_set.shape)
print(theta_dot_set.shape)
print(velocoty_theta_dot_set.shape)
print(velocoty_theta_2dot_set.shape)

fig, ax = plt.subplots()
theta_n = int(theta_set.size/20)
theta_dot_n = int(theta_dot_set.size/20)
fig_theta_set = theta_set[::theta_n]
fig_theta_dot_set = theta_dot_set[::theta_dot_n]
fig_velocoty_theta_dot_set = velocoty_theta_dot_set[::theta_n, ::theta_dot_n].T
fig_velocoty_theta_2dot_set = velocoty_theta_2dot_set[::theta_n, ::theta_dot_n].T


print(fig_theta_set.shape)
print(fig_theta_dot_set.shape)
print(fig_velocoty_theta_dot_set.shape)
print(fig_velocoty_theta_2dot_set.shape)

q = ax.quiver(fig_theta_set, fig_theta_dot_set, fig_velocoty_theta_dot_set, fig_velocoty_theta_2dot_set)
ax.quiverkey(q, X=0.3, Y=1.1, U=10, label='Quiver key, length = 10', labelpos='E')

plt.show()
# exit()

def is_nera(val1, val2, wid):
    return abs(val1 - val2) <= wid/2

def is_target_element(val1, val2):
    target_theta = 0
    target_theta_dot = 0
    return is_nera(val1, target_theta, step_theta) and is_nera(val2, target_theta_dot, step_theta_dot)

toropogical_space_concentration = np.array([[1.0 if is_target_element(theta, theta_dot) else 0.0 for theta_dot in theta_dot_set] for theta in theta_set])

print(toropogical_space_concentration)
print("toropogical_space_concentration", toropogical_space_concentration.shape)


def show_plot():
    plt.imshow(toropogical_space_concentration.T, cmap='Blues')
    # plt.xticks(theta_set.tolist())
    # plt.yticks(theta_dot_set.tolist())

    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)
    plt.show()

show_plot()
# exit()

du = 0.1
min_u = -1 * 2
max_u = 1 * 2 * du
u_set = np.arange(min_u, max_u, du)
p_u = 1/u_set.size
dt = 0.01
d_i = np.array([step_theta, step_theta_dot]) * dt * p_u

toropogical_space_velocity = np.array([[[model.singlependulum_dynamics(theta, theta_dot, u) * d_i for u in u_set] for theta_dot in theta_dot_set] for theta in theta_set])
print(toropogical_space_velocity)

def uptade_concentration2():
    theta_dist_grad, theta_dot_dist_grad = np.gradient(toropogical_space_concentration, step_theta, step_theta_dot)

    # print(theta_dist_grad.shape)
    # print(theta_dot_dist_grad.shape)

    # print(toropogical_space_concentration.shape)
    # print(theta_set.size)
    # print(theta_dot_set.size)
    # print("theta_dist_grad")
    # print(theta_dist_grad.shape)

    for x1 in range(1, theta_set.size - 1):
        for x2 in range(1, theta_dot_set.size - 1):
            def possible_variable(v, del_rho):
                x_1 = x1
                x_2 = x2
                if v[0] < 0:
                    x_1 += 1
                else:
                    x_1 -= 1
                if v[1] < 0:
                    x_2 += 1
                else:
                    x_2 -= 1
                return toropogical_space_concentration[x_1, x_2] < del_rho

                
            # print(x1, x2)
            if theta_dist_grad[x1, x2] == 0 and theta_dot_dist_grad[x1, x2] == 0:
                continue
            grad = np.array([theta_dist_grad[x1, x2], theta_dot_dist_grad[x1, x2]])
            velocity_list = toropogical_space_velocity[x1, x2]
            d_rho = 0.
            for velocity in velocity_list:
                d = np.dot(velocity, grad)
                if possible_variable(velocity, d):
                    d_rho += d
            
            # print("update toropogical_space_concentration")
            # print(toropogical_space_velocity[x1, x2])
            # print("s")
            # print(grad)
            # print("s")
            # print(np.sum(np.dot(toropogical_space_velocity[x1, x2], grad)))
            toropogical_space_concentration[x1, x2] += d_rho

            if is_target_element(theta_set[x1], theta_dot_set[x2]):
                toropogical_space_concentration[x1, x2] = 1
            # if toropogical_space_concentration[x1, x2] < 0:
                # print("update toropogical_space_concentration")
                # print(toropogical_space_velocity[x1, x2])
                # print("grad")
                # print(grad)
                # print("s")
                # print(np.sum(np.dot(toropogical_space_velocity[x1, x2], grad)))
                # print("theta {} theta_dot {}".format(theta_set[x1], theta_dot_set[x2]))
                # toropogical_space_concentration[x1, x2] = 0
            if x1 == 0 or x2 == 0 or x1 == theta_set.size - 1 or x2 == theta_dot_set.size - 1:
                toropogical_space_concentration[x1, x2] = 0

for n in tqdm(range(100000)):
    uptade_concentration2()
    if n % 1000 == 0:
        # print(toropogical_space_concentration)
        # print("toropogical_space_concentration", toropogical_space_concentration.shape)
        show_plot()

show_plot()


def uptade_concentration():
    current_concentration = toropogical_space_concentration

    theta_dist_grad, theta_dot_dist_grad = np.gradient(toropogical_space_concentration, step_theta, step_theta_dot)

    # print(theta_dist_grad.shape)
    # print(theta_dot_dist_grad.shape)

    # print(toropogical_space_concentration.shape)

    def update(theta, theta_dot, grad_x1, grad_x2):
        grad = np.array([grad_x1, grad_x2])
        du = 0.1
        dt = 0.0001
        d_rho = 0.

        u = -1
        while(u < 1):
            d_f = np.dot(model.singlependulum_dynamics(theta, theta_dot, u), grad) * f_u(u)
            d_rho += d_f * du
            u += du

        return d_rho * dt


    def f_u(u):
        if -1 < u and u < 1:
            return 0.5
        else:
            return 0

    # print(theta_set.size)
    # print(theta_dot_set.size)
    # print("theta_dist_grad")
    # print(theta_dist_grad.shape)

    for x1 in range(theta_set.size):
        for x2 in range(theta_dot_set.size):
            # print(x1, x2)
            if theta_dist_grad[x1, x2] == 0 and theta_dot_dist_grad[x1, x2] == 0:
                continue
            toropogical_space_concentration[x1, x2] += update(theta_set[x1], theta_dot_set[x2], theta_dist_grad[x1, x2], theta_dot_dist_grad[x1, x2])
            if is_target_element(theta_set[x1], theta_dot_set[x2]):
                toropogical_space_concentration[x1, x2] = 1
            if toropogical_space_concentration[x1, x2] < 0.00000000000001:
                toropogical_space_concentration[x1, x2] = 0
            if x1 == 0 or x2 == 0 or x1 == theta_set.size - 1 or x2 == theta_dot_set.size - 1:
                toropogical_space_concentration[x1, x2] = 0
