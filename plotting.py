import numpy as np

import matplotlib.pyplot as plt
import nbfigtulz as ftl


@ftl.with_context
def plot_error(*, q, p, m, file_name="error"):
    T = -np.sum(m / np.linalg.norm(q, axis=-1), axis=-1)
    V = np.sum(np.sum(p**2, axis=-1) / m, axis=-1) / 2.0
    L = np.sum(q[..., 0] * p[..., 1] - q[..., 1] * p[..., 0], axis=-1)
    H = T + V

    H0 = H[0]
    L0 = L[0]

    fig, ax = plt.subplots()
    ax.set_xlabel("Time")
    ax.set_ylabel("Error")

    ax.plot(H - H0, label="$H(t) - H(0)$")
    ax.plot(L - L0, label="$L_z(t) - L_z(0)")

    ax.legend()

    return ftl.save_fig(fig, file_name)


def render_trj_frame(ax, *, q, m, n):
    ax.clear()

    ax.set_aspect("equal")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    x_min = np.min(q[:, :, 0])
    x_max = np.max(q[:, :, 0])
    y_min = np.min(q[:, :, 1])
    y_max = np.max(q[:, :, 1])

    margin = 0.1
    dx = abs(x_max - x_min) * margin
    dy = abs(y_max - y_min) * margin

    ax.set_xlim(x_min - dx, x_max + dx)
    ax.set_ylim(y_min - dy, y_max + dy)

    q = q[:n]
    q1, q2 = q[:, 0], q[:, 1]
    m1, m2 = m
    Q = (m1 * q1 + m2 * q2) / (m1 + m2)

    ax.plot(0.0, 0.0, "x", c="black")

    ax.plot(q1[:, 0], q1[:, 1], alpha=0.3, linewidth=1)
    ax.plot(q2[:, 0], q2[:, 1], alpha=0.3, linewidth=1)
    ax.plot(Q[:, 0], Q[:, 1], "k-", alpha=0.7)
    
    ax.plot([q1[-1, 0], q2[-1, 0]], [q1[-1, 1], q2[-1, 1]], "k-", alpha=0.5)
    ax.plot(q1[-1, 0], q1[-1, 1], "o", color="C0")
    ax.plot(q2[-1, 0], q2[-1, 1], "o", color="C1")

    ax.plot([q1[0, 0], q2[0, 0]], [q1[0, 1], q2[0, 1]], "k-", alpha=0.5)
    ax.plot(q1[0, 0], q1[0, 1], "o", color="C0")
    ax.plot(q2[0, 0], q2[0, 1], "o", color="C1")

    return ax
