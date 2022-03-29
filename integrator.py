import numba
import numpy as np
import composition


@numba.njit
def force(q, m):
    r3 = np.hypot(q[:, 0], q[:, 1]) ** 3
    return -(m / r3).reshape(-1, 1) * q


@numba.njit
def strang1(qp, m, t, l2):
    q1, q2 = qp[0]
    p1, p2 = qp[1]
    m1, m2 = m

    v0 = (p1 + p2) / (m1 + m2)
    mu = m1 * m2 / (m1 + m2)

    dqx, dqy = q1 - q2
    dvx, dvy = p1 / m1 - p2 / m2
    w = (dvx * dqy - dvy * dqx) / l2

    s = np.sin(w * t)
    omc = 2.0 * np.sin(w * t / 2) ** 2  # 1 - cos(wt)

    delta_q = np.array([dqx * omc - dqy * s, dqy * omc + dqx * s])
    delta_p = np.array([dqy * omc + dqx * s, -dqx * omc + dqy * s])

    return np.stack(
        (
            v0.reshape(1, -1) * t - mu / m1 * np.vstack((delta_q, -delta_q)),
            -mu * w * np.vstack((delta_p, -delta_p)),
        ),
        axis=0,
    )


@numba.njit
def strang2(qp, m, t, l2):
    q1, q2 = qp[0]
    m1, m2 = m

    f1, f2 = force(q=qp[0], m=m)
    dq = q1 - q2

    lmbd = np.dot(m2 * f1 - m1 * f2, dq) / (m1 + m2) / l2

    a1 = f1 - lmbd * dq
    a2 = f2 + lmbd * dq

    return np.stack((np.zeros_like(qp[0]), np.vstack((a1, a2)) * t), axis=0)


@numba.njit
def safe_acc(x, dx, error):
    # Compensated summation
    # by Higham (1993), DOI:10.1137/0914050
    error = error + dx
    x_new = x + error
    error = error + (x - x_new)
    return x_new, error


@numba.njit
def integration_step(qp, error, m, t, l2, gamma):
    dqp = strang1(qp=qp, m=m, t=gamma[0] * t / 2.0, l2=l2)
    qp, error = safe_acc(qp, dqp, error)

    n = len(gamma)
    for i in range(n):
        g2 = gamma[i]
        g1 = g2 + (gamma[i + 1] if i + 1 < n else 0.0)

        dqp = strang2(qp=qp, m=m, t=g2 * t, l2=l2)
        qp, error = safe_acc(qp, dqp, error)

        dqp = strang1(qp=qp, m=m, t=g1 * t / 2.0, l2=l2)
        qp, error = safe_acc(qp, dqp, error)

    return qp, error


@numba.njit
def _integration_loop(qp, m, t, n, l2, gamma):
    error = np.zeros_like(qp)
    for _ in range(n):
        qp, error = integration_step(qp=qp, error=error, m=m, t=t, l2=l2, gamma=gamma)

    return qp


def integration_loop(*, q, p, m, h, n=1, gamma=composition.gamma_p6s9):
    dq = q[0] - q[1]
    l2 = np.dot(dq, dq)

    qp = np.stack((q, p), axis=0)
    qp = _integration_loop(qp, m, h, n, l2, gamma)

    return qp[0], qp[1]
