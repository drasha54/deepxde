import glob
import numpy as np
import deepxde as dde


# Problem constants
a = 1
d = 1
Re = 1


def pde(x, u):
    u_vel, v_vel, w_vel, p = u[:, 0:1], u[:, 1:2], u[:, 2:3], u[:, 3:4]

    u_vel_x = dde.grad.jacobian(u, x, i=0, j=0)
    u_vel_y = dde.grad.jacobian(u, x, i=0, j=1)
    u_vel_z = dde.grad.jacobian(u, x, i=0, j=2)
    u_vel_t = dde.grad.jacobian(u, x, i=0, j=3)
    u_vel_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
    u_vel_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)
    u_vel_zz = dde.grad.hessian(u, x, component=0, i=2, j=2)

    v_vel_x = dde.grad.jacobian(u, x, i=1, j=0)
    v_vel_y = dde.grad.jacobian(u, x, i=1, j=1)
    v_vel_z = dde.grad.jacobian(u, x, i=1, j=2)
    v_vel_t = dde.grad.jacobian(u, x, i=1, j=3)
    v_vel_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)
    v_vel_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)
    v_vel_zz = dde.grad.hessian(u, x, component=1, i=2, j=2)

    w_vel_x = dde.grad.jacobian(u, x, i=2, j=0)
    w_vel_y = dde.grad.jacobian(u, x, i=2, j=1)
    w_vel_z = dde.grad.jacobian(u, x, i=2, j=2)
    w_vel_t = dde.grad.jacobian(u, x, i=2, j=3)
    w_vel_xx = dde.grad.hessian(u, x, component=2, i=0, j=0)
    w_vel_yy = dde.grad.hessian(u, x, component=2, i=1, j=1)
    w_vel_zz = dde.grad.hessian(u, x, component=2, i=2, j=2)

    p_x = dde.grad.jacobian(u, x, i=3, j=0)
    p_y = dde.grad.jacobian(u, x, i=3, j=1)
    p_z = dde.grad.jacobian(u, x, i=3, j=2)

    momentum_x = (
        u_vel_t
        + (u_vel * u_vel_x + v_vel * u_vel_y + w_vel * u_vel_z)
        + p_x
        - 1 / Re * (u_vel_xx + u_vel_yy + u_vel_zz)
    )
    momentum_y = (
        v_vel_t
        + (u_vel * v_vel_x + v_vel * v_vel_y + w_vel * v_vel_z)
        + p_y
        - 1 / Re * (v_vel_xx + v_vel_yy + v_vel_zz)
    )
    momentum_z = (
        w_vel_t
        + (u_vel * w_vel_x + v_vel * w_vel_y + w_vel * w_vel_z)
        + p_z
        - 1 / Re * (w_vel_xx + w_vel_yy + w_vel_zz)
    )
    continuity = u_vel_x + v_vel_y + w_vel_z

    return [momentum_x, momentum_y, momentum_z, continuity]


def u_func(x):
    return (
        -a
        * (
            np.exp(a * x[:, 0:1]) * np.sin(a * x[:, 1:2] + d * x[:, 2:3])
            + np.exp(a * x[:, 2:3]) * np.cos(a * x[:, 0:1] + d * x[:, 1:2])
        )
        * np.exp(-(d ** 2) * x[:, 3:4])
    )


def v_func(x):
    return (
        -a
        * (
            np.exp(a * x[:, 1:2]) * np.sin(a * x[:, 2:3] + d * x[:, 0:1])
            + np.exp(a * x[:, 0:1]) * np.cos(a * x[:, 1:2] + d * x[:, 2:3])
        )
        * np.exp(-(d ** 2) * x[:, 3:4])
    )


def w_func(x):
    return (
        -a
        * (
            np.exp(a * x[:, 2:3]) * np.sin(a * x[:, 0:1] + d * x[:, 1:2])
            + np.exp(a * x[:, 1:2]) * np.cos(a * x[:, 2:3] + d * x[:, 0:1])
        )
        * np.exp(-(d ** 2) * x[:, 3:4])
    )


def p_func(x):
    return (
        -0.5
        * a ** 2
        * (
            np.exp(2 * a * x[:, 0:1])
            + np.exp(2 * a * x[:, 1:2])
            + np.exp(2 * a * x[:, 2:3])
            + 2
            * np.sin(a * x[:, 0:1] + d * x[:, 1:2])
            * np.cos(a * x[:, 2:3] + d * x[:, 0:1])
            * np.exp(a * (x[:, 1:2] + x[:, 2:3]))
            + 2
            * np.sin(a * x[:, 1:2] + d * x[:, 2:3])
            * np.cos(a * x[:, 0:1] + d * x[:, 1:2])
            * np.exp(a * (x[:, 2:3] + x[:, 0:1]))
            + 2
            * np.sin(a * x[:, 2:3] + d * x[:, 0:1])
            * np.cos(a * x[:, 1:2] + d * x[:, 2:3])
            * np.exp(a * (x[:, 0:1] + x[:, 1:2]))
        )
        * np.exp(-2 * d ** 2 * x[:, 3:4])
    )


spatial_domain = dde.geometry.Cuboid(xmin=[-1, -1, -1], xmax=[1, 1, 1])
temporal_domain = dde.geometry.TimeDomain(0, 1)
spatio_temporal_domain = dde.geometry.GeometryXTime(spatial_domain, temporal_domain)

boundary_condition_u = dde.icbc.DirichletBC(
    spatio_temporal_domain, u_func, lambda _, on_boundary: on_boundary, component=0
)
boundary_condition_v = dde.icbc.DirichletBC(
    spatio_temporal_domain, v_func, lambda _, on_boundary: on_boundary, component=1
)
boundary_condition_w = dde.icbc.DirichletBC(
    spatio_temporal_domain, w_func, lambda _, on_boundary: on_boundary, component=2
)

initial_condition_u = dde.icbc.IC(
    spatio_temporal_domain, u_func, lambda _, on_initial: on_initial, component=0
)
initial_condition_v = dde.icbc.IC(
    spatio_temporal_domain, v_func, lambda _, on_initial: on_initial, component=1
)
initial_condition_w = dde.icbc.IC(
    spatio_temporal_domain, w_func, lambda _, on_initial: on_initial, component=2
)

def load_model():
    data = dde.data.TimePDE(
        spatio_temporal_domain,
        pde,
        [
            boundary_condition_u,
            boundary_condition_v,
            boundary_condition_w,
            initial_condition_u,
            initial_condition_v,
            initial_condition_w,
        ],
        num_domain=10,
        num_boundary=2,
        num_initial=2,
        num_test=10,
    )

    net = dde.nn.FNN([4] + 4 * [50] + [4], "tanh", "Glorot normal")
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3)
    path = sorted(glob.glob("Beltrami_flow_model*"))[-1]
    model.restore(path, verbose=1)
    return model

def main():
    model = load_model()
    x, y, z = np.meshgrid(
        np.linspace(-1, 1, 10), np.linspace(-1, 1, 10), np.linspace(-1, 1, 10)
    )
    X = np.vstack((np.ravel(x), np.ravel(y), np.ravel(z))).T
    t = np.zeros((X.shape[0], 1))
    X_input = np.hstack((X, t))
    preds = model.predict(X_input)
    print(preds[:5])


if __name__ == "__main__":
    main()
