import numpy as np
import jax
import jax.numpy as jnp
import tensorflow as tf
import matplotlib.pyplot as plt

from rendering import render_frame


def fig2rgb_array(fig, expand=True):
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    shape = (nrows, ncols, 3) if not expand else (1, nrows, ncols, 3)
    return np.fromstring(buf, dtype=np.uint8).reshape(shape)


def mpl_image(callback):
    fig = plt.figure(frameon=False, figsize=(8, 8), dpi=128)
    ax = fig.add_axes([0, 0, 1, 1])
    callback()
    plt.axis("off")
    fig.patch.set_visible(False)
    ax.patch.set_visible(False)
    image = fig2rgb_array(fig)
    plt.close(fig)
    return image


def circles(s, *args, **kwargs):
    s = jnp.reshape(s, (-1, 3))
    theta = jnp.linspace(0, 2 * jnp.pi, 64)
    x = jnp.reshape(jnp.cos(theta), (1, -1))
    y = jnp.reshape(jnp.sin(theta), (1, -1))
    x = jnp.transpose(x * s[:, [2]] + s[:, [1]])
    y = jnp.transpose(y * s[:, [2]] + s[:, [0]])
    return plt.plot(x, y, *args, **kwargs)[0:len(s)]


def visualizations_2d(params, model_fn, step):
    image_dim = 1024

    i = jnp.linspace(-0.5, 0.5, image_dim)
    j = jnp.linspace(-0.5, 0.5, image_dim)
    xy = jnp.stack(jnp.meshgrid(i, j, indexing="ij"), axis=-1)
    xy_flat = xy.reshape(-1, 2)

    sdf = []
    agrad = []
    pgrad = []
    mf = []

    batch_size = 8192
    for n in jnp.arange(xy_flat.shape[0] // batch_size):
        xy_batch = xy_flat[batch_size * n:batch_size * (n + 1)]
        sdf_n, agrad_n, pgrad_n, mf_n = model_fn(params, xy_batch)
        sdf.append(sdf_n)
        agrad.append(agrad_n)
        pgrad.append(pgrad_n)
        mf.append(mf_n)

    sdf = jnp.concatenate(sdf, axis=0).reshape(image_dim, image_dim, 1)
    agrad = jnp.concatenate(agrad, axis=0).reshape(image_dim, image_dim, 2)
    pgrad = jnp.concatenate(pgrad, axis=0).reshape(image_dim, image_dim, 2)
    mf = jnp.concatenate(mf, axis=0).reshape(image_dim, image_dim, 1)

    def sdf_vis():
        plt.imshow(sdf)
        plt.contour(sdf[..., 0], levels=[0.0], colors="black")

    tf.summary.image("Predicted SDF", mpl_image(sdf_vis), step)

    agradc = (agrad + 1) / 2
    agradc = jnp.concatenate([agradc, jnp.ones_like(agrad[..., 0:1])], axis=-1)
    agradc = jnp.clip(agradc, 0., 1.)

    def agrad_vis():
        plt.imshow(agradc)

    tf.summary.image("Analytic Gradient", mpl_image(agrad_vis), step)

    pgradc = (pgrad + 1) / 2
    pgradc = jnp.concatenate([pgradc, jnp.ones_like(pgrad[..., 0:1])], axis=-1)
    pgradc = jnp.clip(pgradc, 0., 1.)

    def pgrad_vis():
        plt.imshow(pgradc)

    tf.summary.image("Predicted Gradient", mpl_image(pgrad_vis), step)

    def mf_vis():
        plt.imshow(mf)

    tf.summary.image("Predicted Medial Field", mpl_image(mf_vis), step)

    agrad_norm = jnp.linalg.norm(agrad, axis=-1, keepdims=True)
    ac_dir = jnp.sign(sdf) * agrad / agrad_norm
    ac = (mf - jnp.abs(sdf)) * ac_dir + xy
    ac_histogram, _, _ = jnp.histogram2d(ac[..., 0].reshape(-1),
                                         ac[..., 1].reshape(-1),
                                         jnp.linspace(-0.5, 0.5, 1024))

    def ac_vis():
        plt.imshow(jnp.log(ac_histogram + 0.01))
        plt.contour(sdf[..., 0], levels=[0.0], colors="black")

    tf.summary.image("Centers (Analytic Grad)", mpl_image(ac_vis), step)

    pgrad_norm = jnp.linalg.norm(pgrad, axis=-1, keepdims=True)
    pc_dir = jnp.sign(sdf) * pgrad / pgrad_norm
    pc = (mf - jnp.abs(sdf)) * pc_dir + xy
    pc_histogram, _, _ = jnp.histogram2d(pc[..., 0].reshape(-1),
                                         pc[..., 1].reshape(-1),
                                         jnp.linspace(-0.5, 0.5, 1024))

    def pc_vis():
        plt.imshow(jnp.log(pc_histogram + 0.01))
        plt.contour(sdf[..., 0], levels=[0.0], colors="black")

    tf.summary.image("Centers (Predicted Grad)", mpl_image(pc_vis), step)

    c_sub = ac[::32, ::32]
    mf_sub = mf[::32, ::32]
    sdf_sub = sdf[::32, ::32]

    s = jnp.concatenate([c_sub, mf_sub], axis=-1).reshape(-1, 3)
    s_sdf = sdf_sub.reshape(-1)
    s_inner = s[s_sdf < 0.0]
    s_outer = s[s_sdf >= 0.0]

    def circles_vis():
        circles(s_inner, "r", linewidth=0.5)
        circles(s_outer, "b", linewidth=0.5)
        plt.xlim(-0.5, 0.5)
        plt.ylim(0.5, -0.5)

    tf.summary.image("Medial Spheres", mpl_image(circles_vis), step)


def visualizations_3d(params, model_fn, step):
    point, grad, its, depth = render_frame(model_fn, params)

    grad_col = grad / jnp.linalg.norm(grad, axis=-1, keepdims=True)
    grad_col = 0.5 * grad_col + 0.5
    grad_col *= (depth < 4.0)[:, :, None]

    tf.summary.image("Normals", grad_col[None], step)


def get_summary_fn(summary_path, model):
    writer = tf.summary.create_file_writer(summary_path)

    @jax.jit
    def model_fn(params, x):
        return model.apply(params, x, method=model.with_sdf_grad)

    def summary_fn(params, step, loss_values):
        loss_mean = {}
        for k in loss_values[0]:
            loss_mean[k] = jnp.mean(jnp.stack(list(d[k] for d in loss_values)))

        with writer.as_default():
            with tf.name_scope("Losses"):
                for k in loss_mean:
                    tf.summary.scalar(k, loss_mean[k], step)

            with tf.name_scope("Visualizations"):
                if model.dimensions == 2:
                    visualizations_2d(params, model_fn, step)

                elif model.dimensions == 3:
                    visualizations_3d(params, model_fn, step)

    return summary_fn