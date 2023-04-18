import os

import jax
import jax.numpy as jnp
import skimage.io
import skimage.morphology
import scipy.ndimage.morphology as morph
import trimesh

from bilinear import sample_bilinear

scenes_2d = set(["octopus", "koch", "giraffe", "m", "maple", "statue"])


def load_scene_2d(name, data_directory, batch_size):
    image_path = os.path.join(data_directory, name + ".png")
    im = jax.device_put(skimage.io.imread(image_path)[:, :, 0])
    im = jax.image.resize(im, (4096, 4096), method="bilinear")

    max_dim = max(im.shape[0], im.shape[1])

    dt_out = morph.distance_transform_edt(im > 128)
    dt_in = morph.distance_transform_edt(im <= 128)
    sdf_gt = (dt_out - dt_in) / max_dim

    def sdf_oracle(p):
        return sample_bilinear(sdf_gt[None], p[None] + 0.5, True)[0]

    edge_grad_x, edge_grad_y = jnp.gradient(im)
    edge_grad_x = skimage.filters.gaussian(edge_grad_x, sigma=5.0)
    edge_grad_y = skimage.filters.gaussian(edge_grad_y, sigma=5.0)
    edge_grad_gt = jnp.stack([edge_grad_x, edge_grad_y], axis=-1)

    edges = jnp.logical_xor(skimage.morphology.dilation(im > 128), im > 128)

    edge_xy = jnp.stack(jnp.nonzero(edges), axis=-1).astype(jnp.float32)
    edge_xy = edge_xy / max_dim - 0.5
    edge_inds = jnp.stack(jnp.nonzero(edges.reshape(-1)), axis=-1)

    edge_grad_gt = edge_grad_gt.reshape(-1, 2)[edge_inds[:, 0]]
    edge_grad_gt /= jnp.linalg.norm(edge_grad_gt, axis=-1, keepdims=True)

    edge_sdf_gt = sdf_gt.reshape(-1)[edge_inds]
    edge_xy -= edge_grad_gt * edge_sdf_gt

    n_edges = edge_xy.shape[0]

    @jax.jit
    def sample(k):
        k0, k1 = jax.random.split(k, 2)
        inds = jax.random.randint(k0, (batch_size, ), 0, n_edges)
        zero_set_xy = edge_xy[inds]
        zero_set_grad_gt = edge_grad_gt[inds]

        offsets = 0.5 * jax.random.normal(k1, zero_set_xy.shape)
        area_xy = zero_set_xy + offsets
        area_sdf_gt = sdf_oracle(area_xy)

        return zero_set_xy, zero_set_grad_gt, area_xy, area_sdf_gt

    k = jax.random.PRNGKey(0)
    while True:
        k, k_i = jax.random.split(k)
        yield sample(k_i)


scenes_3d = set(["mecha", "armadillo", "bunny", "horse", "lucy", "rocker-arm"])


def load_scene_3d(name, data_directory, batch_size, n_samples=2**23):
    mesh_path = os.path.join(data_directory, name + ".obj")
    mesh = trimesh.load_mesh(mesh_path)

    def sample_oriented(mesh, N):
        point, f_index = trimesh.sample.sample_surface(mesh, N)

        v_index = mesh.faces[f_index]
        v_pos = mesh.vertices[v_index]
        v_normal = mesh.vertex_normals[v_index]

        baryc = trimesh.triangles.points_to_barycentric(v_pos, point)
        p_normal = jnp.sum(baryc[..., None] * v_normal, axis=1)
        p_normal /= jnp.linalg.norm(p_normal, axis=-1, keepdims=True)

        return point, p_normal

    x_gt, n_gt = sample_oriented(mesh, n_samples)

    x_gt -= jnp.median(x_gt, axis=0, keepdims=True)
    diag = (jnp.max(x_gt, axis=0, keepdims=True) -
            jnp.min(x_gt, axis=0, keepdims=True))
    x_gt /= jnp.max(diag)
    x_gt, n_gt = (jax.device_put(x_gt), jax.device_put(n_gt))

    @jax.jit
    def sample(k):
        k0, k1 = jax.random.split(k, 2)
        inds = jax.random.randint(k0, (batch_size, ), 0, n_samples)
        zero_set_x = x_gt[inds]
        zero_set_grad_gt = n_gt[inds]

        offsets = 0.5 * jax.random.normal(k1, zero_set_x.shape)
        area_x = zero_set_x + offsets
        area_sdf_gt = None

        return zero_set_x, zero_set_grad_gt, area_x, area_sdf_gt

    k = jax.random.PRNGKey(0)
    while True:
        k, k_i = jax.random.split(k)
        yield sample(k_i)


def get_dataset(name, data_directory, batch_size):
    if name in scenes_2d:
        iterator = load_scene_2d(name, data_directory, batch_size)
        dimensions = 2
    elif name in scenes_3d:
        iterator = load_scene_3d(name, data_directory, batch_size)
        dimensions = 3
    else:
        raise ValueError("Unknown Dataset '{}'".format(name))

    return iterator, dimensions
