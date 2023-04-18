import os

import jax

from config import get_args
from data import get_dataset
from loss import compute_losses
from model import build_and_restore_model
from summary import get_summary_fn
from training import train


def main(args):
    experiment_path = os.path.join(args.experiments_dir, args.experiment_name)
    checkpoint_path = os.path.join(experiment_path, "checkpoint")
    summary_path = os.path.join(experiment_path, "summary")
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(summary_path, exist_ok=True)

    data_iterator, dimensions = get_dataset(args.dataset, args.datasets_dir,
                                            args.batch_size)

    model, params, opt_state, step = build_and_restore_model(dimensions, args,
                                                             checkpoint_path,
                                                             args.model_seed)

    def loss_fn(params, k, data, t):
        def model_fn(x): return model.apply(
            params, x, method=model.with_mf_grad)
        return compute_losses(model_fn, k, data, t, args.weights)

    summary_fn = get_summary_fn(summary_path, model)

    k_train = jax.random.PRNGKey(1)

    print("Beginning training")
    train(params, opt_state, args.learning_rate, loss_fn, k_train, data_iterator, summary_fn,
          checkpoint_path, args.total_iterations, step, args.summary_period,
          args.checkpoint_period, args.keep_checkpoints)


if __name__ == "__main__":
    main(get_args())
