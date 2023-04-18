import os

import jax
import matplotlib.pyplot as plt
import jax.numpy as jnp
import scipy.ndimage
import numpy as np
import skimage.transform

from config import get_args
from data import get_dataset
from model import build_and_restore_model
from rendering import render_frame


def main(args):
    experiment_path = os.path.join(args.experiments_dir, args.experiment_name)
    checkpoint_path = os.path.join(experiment_path, "checkpoint")
    summary_path = os.path.join(experiment_path, "summary")
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(summary_path, exist_ok=True)

    _, dimensions = get_dataset(args.dataset, args.datasets_dir,
                                args.batch_size)

    model, params, _, _ = build_and_restore_model(dimensions, args,
                                                  checkpoint_path,
                                                  args.model_seed)

    @jax.jit
    def model_fn(params, p):
        return model.apply(params, p, method=model.with_sdf_grad)

    cameras = {
        "armadillo": (jnp.array([0.0, -0.1, -0.9]), jnp.array([0.0, 0.0, 1.0]),
                      jnp.array([0.0, 1.0, 0.0])),
        "bunny": (jnp.array([0.05, 0.0, 1.0]), jnp.array([0.0, 0.0, -1.0]),
                  jnp.array([0.0, 1.0, 0.0])),
        "horse": (jnp.array([-0.9, 0.05, 0.0]), jnp.array([1.0, 0.0, 0.0]),
                  jnp.array([0.0, 0.0, 1.0])),
        "lucy": (jnp.array([0.0, 0.9, 0.05]), jnp.array([0.0, -1.0, 0.0]),
                 jnp.array([0.0, 0.0, 1.0])),
        "mecha": (jnp.array([0.0, -0.85, -0.1]), jnp.array([0.0, 1.0, 0.0]),
                  jnp.array([0.0, 0.0, 1.0])),
        "rocker-arm": (jnp.array([-0.8, 0.0, 0.0]), jnp.array([1.0, 0.0, 0.0]),
                       jnp.array([0.0, 0.0, 1.0])),
    }

    eye, forward, up = cameras[args.dataset]

    res = 256
    point, grad, _, depth = render_frame(model_fn, params, eye, forward, up,
                                         1.3, res)

    psize = 61

    def ssao_filt(patch):
        center = patch[(psize * psize) // 2]
        return 1.0 - (np.sum((patch < center) * 1.0) / psize**2 - 0.5)

    ssao = scipy.ndimage.generic_filter(depth, ssao_filt, (psize, psize))
    ssao = ssao.reshape(res, res, 1)
    ssao = jnp.clip(ssao, 0.0, 1.0)

    off_point = point + 0.01 * grad
    _, _, _, mf = model_fn(params, off_point.reshape(-1, 3))
    mf = mf.reshape(res, res, 1)

    mf = jnp.clip(mf, 0.0, 1e7)
    ao = 2.5 * (mf + 0.005)**0.3
    ao = jnp.clip(ao, 0.05, 1.0)

    light = forward + -up
    light = -light / jnp.linalg.norm(light)

    factor = jnp.sum(light[None, None] * grad, axis=-1, keepdims=True)
    factor = jnp.clip(factor, 0.0, 1.0)

    color = jnp.zeros_like(grad) + factor
    color = 0.3 * color + 0.7
    color *= (depth < 4.0)[:, :, None]

    alpha = (depth < 4.0)[:, :, None] + jnp.zeros_like(factor)

    rgba = jnp.concatenate([color, alpha], axis=-1)
    rgba = skimage.transform.resize(rgba, (768, 768), anti_aliasing=True)
    skimage.io.imsave("{}_noao.png".format(args.dataset), rgba)

    color = jnp.zeros_like(grad) + factor
    color = 0.3 * color + 0.7 * ao
    color *= (depth < 4.0)[:, :, None]

    alpha = (depth < 4.0)[:, :, None] + jnp.zeros_like(factor)

    rgba = jnp.concatenate([color, alpha], axis=-1)
    rgba = skimage.transform.resize(rgba, (768, 768), anti_aliasing=True)
    skimage.io.imsave("{}_mfao.png".format(args.dataset), rgba)

    color = jnp.zeros_like(grad) + factor
    color = 0.3 * color + 0.7 * ssao
    color *= (depth < 4.0)[:, :, None]

    alpha = (depth < 4.0)[:, :, None] + jnp.zeros_like(factor)

    rgba = jnp.concatenate([color, alpha], axis=-1)
    rgba = skimage.transform.resize(rgba, (768, 768), anti_aliasing=True)
    skimage.io.imsave("{}_ssao.png".format(args.dataset), rgba)

    def grad_fn(x):
        _, grad, _, _ = model_fn(params, x)
        return jnp.sum(grad, axis=0)

    curv = jax.jacrev(grad_fn)(off_point.reshape(-1, 3))
    curv = curv**2
    curv = jnp.sum(curv, axis=-1)
    curv = jnp.sum(curv, axis=0)
    curv = jnp.sqrt(curv)
    curv = curv.reshape(res, res, 1)
    sdfao = jnp.exp(-curv / 1000.0)

    color = jnp.zeros_like(grad) + factor
    color = 0.3 * color + 0.7 * sdfao
    color *= (depth < 4.0)[:, :, None]

    alpha = (depth < 4.0)[:, :, None] + jnp.zeros_like(factor)

    rgba = jnp.concatenate([color, alpha], axis=-1)
    rgba = skimage.transform.resize(rgba, (768, 768), anti_aliasing=True)
    skimage.io.imsave("{}_sdfao.png".format(args.dataset), rgba)


if __name__ == "__main__":
    main(get_args())
