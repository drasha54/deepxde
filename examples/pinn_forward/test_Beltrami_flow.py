import glob
import numpy as np
import deepxde as dde
import torch

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
    '''
    try:
        path = sorted(glob.glob("Beltrami_flow_model*"))[-1]
        #model.restore(path, verbose=1)
        model.restore(path, only_weights=True, verbose=1)
    except (IndexError, FileNotFoundError):
        print("Pretrained model not found. Using randomly initialized weights.")
    return model
    '''
    try:
        # examples ディレクトリではなく /home/nagano/deepxde を検索
        candidates = sorted(glob.glob("/home/nagano/deepxde/Beltrami_flow_model-*.pt"))
        if not candidates:
            raise FileNotFoundError("No checkpoint found in /home/nagano/deepxde")
        path = candidates[-1]
        # torch.load で checkpoint を読み込み
        ckpt = torch.load(path, map_location=torch.device("cpu"))
        # 'model_state_dict' のみ取り出してネットワークに適用
        model.net.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded weights from {path}")
    except (IndexError, FileNotFoundError, KeyError) as e:
        print("Pretrained model not found or invalid checkpoint, using random init:", e)
    return model

def main():
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    model = load_model()

    grid_size = 50
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    Xg, Yg = np.meshgrid(x, y)
    #z平面を指定
    z0 = 0.5
    Zg = np.full_like(Xg, z0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    im_exact = axes[0].imshow(np.zeros_like(Xg), extent=[-1, 1, -1, 1], origin="lower")
    axes[0].set_title("Exact u")
    im_pred = axes[1].imshow(np.zeros_like(Xg), extent=[-1, 1, -1, 1], origin="lower")
    axes[1].set_title("Predicted u")
    im_err = axes[2].imshow(np.zeros_like(Xg), extent=[-1, 1, -1, 1], origin="lower")
    axes[2].set_title("|u - u_exact|")

    # Colorbars: one shared by exact and predicted values, and one for the error
    cbar_uv = fig.colorbar(im_exact, ax=axes[:2])
    cbar_uv.set_label("u value")
    cbar_err = fig.colorbar(im_err, ax=axes[2])
    cbar_err.set_label("Error")

    def update(frame):
        t = np.full_like(Xg, frame)
        X_input = np.stack([Xg, Yg, Zg, t], axis=-1).reshape(-1, 4)
        preds = model.predict(X_input)[:, 0].reshape(grid_size, grid_size)
        exact = p_func(X_input).reshape(grid_size, grid_size)
        err = np.abs(preds - exact)
        # Use common color limits for exact and predicted values
        vmin = min(exact.min(), preds.min())
        vmax = max(exact.max(), preds.max())
        im_exact.set_data(exact)
        im_exact.set_clim(-3, 3)
        im_pred.set_data(preds)
        im_pred.set_clim(-3, 3)
        # 誤差マップの表示更新
        im_err.set_data(err)
        im_err.set_clim(0, 3)
        cbar_uv.update_normal(im_exact)
        cbar_err.update_normal(im_err)
        axes[0].set_xlabel(f"t = {frame:.2f}")
        axes[1].set_xlabel(f"t = {frame:.2f}")
        axes[2].set_xlabel(f"t = {frame:.2f}")
        return [im_exact, im_pred, im_err, cbar_uv.ax, cbar_err.ax]

    times = np.linspace(0, 1, 20)
    ani = FuncAnimation(fig, update, frames=times, blit=False)
    ani.save("u_prediction.gif", writer=PillowWriter(fps=2))

    plt.close(fig)


if __name__ == "__main__":
    main()
