import jax
import jax.numpy as jnp


def get_default_weights():
    return {
        "zero_set_sdf": 10000.0,
        "zero_set_grad": 10.0,
        "volume_sdf": 0.0,
        "eikonal": 1.0,
        "predicted_grad": 1.0,
        "maximality": 100.0,
        "inscription": 500.0,
        "orthogonality": 0.03,
        "surface_reg": 1.0,
        "curvature": 1.0,
    }


def compute_losses(model_fn, k, data, t, weights):
    zero_set_x, zero_set_grad_gt, volume_x, volume_sdf_gt = data

    total_loss = 0.0
    loss_dict = {}

    zero_set_sdf, zero_set_agrad, _, _, _ = model_fn(zero_set_x)

    # Zero Set Losses
    if weights["zero_set_sdf"] > 0:
        zero_set_sdf_loss = jnp.mean(zero_set_sdf**2)
        loss_dict["Zero Set SDF"] = zero_set_sdf_loss
        total_loss += weights["zero_set_sdf"] * zero_set_sdf_loss

    if weights["zero_set_grad"] > 0:
        zero_set_grad_loss = jnp.mean((zero_set_agrad - zero_set_grad_gt)**2)
        loss_dict["Zero Set Gradient"] = zero_set_grad_loss
        total_loss += weights["zero_set_grad"] * zero_set_grad_loss

    # Volume Losses
    sdf, agrad, pgrad, mf, mf_grad = model_fn(volume_x)
    agrad_norm = jnp.linalg.norm(agrad, axis=-1, keepdims=True)
    pgrad_norm = jnp.linalg.norm(pgrad, axis=-1, keepdims=True)

    c_dir = jnp.sign(sdf) * agrad / agrad_norm
    c = (jnp.maximum(mf, jnp.abs(sdf)) - jnp.abs(sdf)) * c_dir + volume_x
    c_sdf, _, _, _, _ = model_fn(c)

    if weights["volume_sdf"] > 0:
        volume_sdf_loss = jnp.mean((sdf - volume_sdf_gt)**2)
        loss_dict["Volume SDF"] = volume_sdf_loss
        total_loss += weights["volume_sdf"] * volume_sdf_loss

    if weights["eikonal"] > 0:
        eikonal_loss = jnp.mean((agrad_norm - 1.0)**2)
        eikonal_loss += jnp.mean((pgrad_norm - 1.0)**2)
        loss_dict["Eikonal"] = eikonal_loss
        total_loss += weights["eikonal"] * eikonal_loss

    if weights["predicted_grad"] > 0:
        predicted_grad_loss = jnp.mean((agrad - pgrad)**2)
        loss_dict["Gradient Prediction"] = predicted_grad_loss
        total_loss += weights["predicted_grad"] * predicted_grad_loss

    if weights["surface_reg"] > 0:
        surface_reg_loss = jnp.mean(jnp.exp(-100.0 * jnp.abs(sdf)))
        loss_dict["Surface Regularizer"] = surface_reg_loss
        total_loss += weights["surface_reg"] * surface_reg_loss

    # Curvature Loss
    curv_x = volume_x[:2048]
    curv_grad = agrad[:2048]

    def f(t):
        _, g, _, _, _ = model_fn(curv_x + curv_grad * t)
        return jnp.sum(g, axis=0)

    curvature = jax.jacrev(f)(jnp.zeros_like(curv_x[..., :1]))[..., 0]

    if weights["curvature"] > 0:
        curvature_loss = jnp.mean(jnp.sum(jnp.abs(curvature), axis=0))
        loss_dict["Curvature"] = curvature_loss
        sched = jnp.power(10, -(1.0 + 4.0 * t))
        total_loss += weights["curvature"] * sched * curvature_loss

    # Medial Field Losses
    if weights["maximality"] > 0:
        maximality_loss = jnp.mean(jax.nn.relu(jnp.abs(sdf) - mf)**2)
        loss_dict["Maximality"] = maximality_loss
        total_loss += weights["maximality"] * maximality_loss

    if weights["inscription"] > 0:
        inscription_loss = jnp.mean((jnp.abs(c_sdf) - mf)**2)
        loss_dict["Inscription"] = inscription_loss
        total_loss += weights["inscription"] * inscription_loss

    if weights["orthogonality"] > 0:
        orthogonality_loss = jnp.mean(jnp.sum(mf_grad * agrad, axis=-1)**2)
        loss_dict["Orthogonality"] = orthogonality_loss
        total_loss += weights["orthogonality"] * orthogonality_loss

    loss_dict["Total"] = total_loss

    return total_loss, loss_dict
