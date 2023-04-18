import jax
from math import ceil, log2
import jax.numpy as jnp


def trace_rays(model_fn,
               rays,
               far,
               eps=1e-4,
               understep=0.95,
               max_iterations=100,
               medial=True):
    dim = rays.shape[-1] // 2
    point = rays[..., :dim]
    dirs = rays[..., dim:]
    dirs /= jnp.linalg.norm(dirs, axis=-1, keepdims=True)

    dist = jnp.zeros_like(point[..., :1])
    iterations = jnp.zeros(rays.shape[:-1], dtype=jnp.int32)

    for i in range(max_iterations):
        sdf, grad, pgrad, mf = model_fn(point)
        converged = jnp.logical_or(jnp.abs(sdf) < eps, dist > far)

        udf = jnp.abs(sdf)

        if medial:
            grad = pgrad
            grad = grad / jnp.linalg.norm(grad, axis=-1, keepdims=True)
            r = mf
            c = (r - udf) * jnp.sign(sdf) * grad + point

        else:
            c = point
            r = udf

        def intersection(c, r):
            alpha = point - c
            beta = -jnp.sum(alpha * dirs, axis=-1, keepdims=True)
            delta = jnp.sqrt(beta**2 - (
                jnp.linalg.norm(alpha, axis=-1, keepdims=True)**2 - r**2))
            i = beta + delta
            i = jnp.where((beta - delta) < 0, i, jnp.zeros_like(i))
            return i

        step = sdf

        if medial:
            istep = intersection(c, r)
            istep = jnp.where(jnp.isnan(istep), jnp.zeros_like(istep), istep)

            step = jnp.maximum(istep, udf)
            step = jnp.where(sdf < 0, sdf, step)

        step = step * understep

        new_point = point + dirs * step
        point = jnp.where(converged, point, new_point)
        dist += step * (1.0 - converged)
        iterations += 1 - converged[..., 0]

    sdf, grad, _, _ = model_fn(point)

    normal = grad / jnp.linalg.norm(grad, axis=-1, keepdims=True)
    intersect = point

    return intersect, normal, iterations


def render_frame(model_fn,
                 params,
                 eye=jnp.array([0.0, 2.0, 0.0]),
                 forward=jnp.array([0.0, -1.0, 0.0]),
                 up=jnp.array([0.0, 0.0, 1.0]),
                 fov=0.7,
                 res=512,
                 medial=False):
    i = jnp.linspace(-1.0, 1.0, res)
    j = jnp.linspace(-1.0, 1.0, res)[::-1]
    I, J = jnp.meshgrid(i, j, indexing="xy")

    right = jnp.cross(forward, up)
    right /= jnp.linalg.norm(right)

    dirs = forward[None, None]
    dirs += right[None, None] * I[..., None] * jnp.tan(fov / 2)
    dirs += up[None, None] * J[..., None] * jnp.tan(fov / 2)
    dirs /= jnp.linalg.norm(dirs, axis=-1, keepdims=True)

    pos = jnp.zeros_like(dirs) + eye[None, None]
    rays = jnp.concatenate([pos, dirs], axis=-1)

    runner = lambda p: model_fn(params, p)

    bs = min(2**18, res * res) // res

    point = []
    grad = []
    its = []
    for i in range(res // bs):
        point_i, grad_i, its_i = trace_rays(runner,
                                            rays[i * bs:(i + 1) * bs],
                                            4.0,
                                            max_iterations=100,
                                            medial=medial)
        point.append(point_i)
        grad.append(grad_i)
        its.append(its_i)

    point = jnp.concatenate(point, axis=0)
    grad = jnp.concatenate(grad, axis=0)
    its = jnp.concatenate(its, axis=0)

    depth = jnp.linalg.norm(point - eye, axis=-1)

    return point, grad, its, depth