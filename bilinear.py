import jax
import jax.numpy as jnp


def sample_bilinear(images, uv, clamp=False):
    # images: N x H x W x C
    # uv: N x S x 2
    if clamp:
        uv = jnp.clip(uv, 0.0, 0.999999)
    image_dims = jnp.array(images.shape[1:3])
    image_dims = image_dims[None, None]

    im_uv = uv * (image_dims - 1)
    min_corner = jnp.floor(im_uv)
    max_corner = min_corner + 1

    corners = jnp.stack([
        min_corner, max_corner,
        jnp.stack((min_corner[..., 0], max_corner[..., 1]), axis=-1),
        jnp.stack((max_corner[..., 0], min_corner[..., 1]), axis=-1)
    ],
                        axis=-2)

    inside = jnp.logical_and(corners >= 0,
                             corners < jnp.expand_dims(image_dims, -2))
    inside = jnp.logical_and(inside[..., 0:1], inside[..., 1:2])

    corner_inds = jnp.where(inside, corners, jnp.zeros_like(corners))
    corner_inds = corner_inds.astype(int)

    batch_inds = jnp.reshape(jnp.arange(images.shape[0]), (-1, 1, 1))

    batch_inds = batch_inds + 0 * corners[..., 0]
    batch_inds = batch_inds.astype(int)

    corner_samples = jnp.array(images)[batch_inds, corner_inds[..., 0],
                                       corner_inds[..., 1]][..., None]
    corner_samples = jnp.where(inside, corner_samples,
                               jnp.zeros_like(corner_samples))

    rel_uv = jnp.expand_dims(im_uv, -2) - corners
    mix = 1.0 - jnp.abs(rel_uv)
    mixed = corner_samples * mix[:, :, :, 0:1] * mix[:, :, :, 1:2]

    val = mixed[:, :, 0] + mixed[:, :, 1] + mixed[:, :, 2] + mixed[:, :, 3]

    return val
