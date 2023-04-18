import jax
import jax.numpy as jnp
import flax.training.checkpoints as checkpoints
import optax
import tqdm


def train(params, opt_state, learning_rate, loss_fn, k, data_iterator, summary_fn, checkpoint_path,
          total_iterations, start_iteration, summary_period, checkpoint_period,
          keep_checkpoints):
    optimizer = optax.adam(learning_rate)

    @jax.jit
    def train_step(k, params, opt_state, data, step):
        t = jnp.array(step, dtype=float) / total_iterations
        def _loss_fn(p): return loss_fn(p, k, data, t)
        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
        (_, loss_dict), grad = grad_fn(params)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return loss_dict, params, opt_state

    loss_values = []
    for step in tqdm.trange(start_iteration, total_iterations, 1):
        data = next(data_iterator)
        k, k_i = jax.random.split(k)
        loss_dict, params, opt_state = train_step(
            k_i, params, opt_state, data, step)
        loss_values.append(loss_dict)

        if (step - start_iteration) != 0 and step % summary_period == 0:
            summary_fn(params, step, loss_values)
            loss_values.clear()

        if (step - start_iteration) != 0 and step % checkpoint_period == 0:
            state = (params, opt_state, step)
            checkpoints.save_checkpoint(checkpoint_path,
                                        state,
                                        step,
                                        keep=keep_checkpoints)
