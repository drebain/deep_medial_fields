import os
import yaml

import configargparse

from loss import get_default_weights


def get_args():
    parser = configargparse.ArgumentParser()

    # Experiment
    parser.add_argument("-n", "--experiment_name", required=True)
    parser.add_argument("--experiments_dir", default="./experiments")

    # Dataset
    parser.add_argument("--datasets_dir", default="./datasets")
    parser.add_argument("-d", "--dataset", required=True)

    # Model
    parser.add_argument("--common_layers", default=6, type=int)
    parser.add_argument("--per_head_layers", default=2, type=int)
    parser.add_argument("--common_layer_units", default=256, type=int)
    parser.add_argument("--per_head_units", default=64, type=int)
    parser.add_argument("--n_fourier_basis", default=64, type=int)
    parser.add_argument("--fourier_freq_sigma", default=16.0, type=float)
    parser.add_argument("--model_seed", default=42, type=int)

    # Losses
    default_weights = get_default_weights()

    def parse_weights(inp):
        weights = dict(default_weights)
        updates = yaml.safe_load(inp)
        for k in updates:
            if k in weights:
                weights[k] = updates[k]
            else:
                raise ValueError("Unknown weight {}".format(k))

        return weights

    parser.add_argument("--weights",
                        default=default_weights,
                        type=parse_weights)

    # Training
    parser.add_argument("-b", "--batch_size", default=2**13, type=int)
    parser.add_argument("-i", "--total_iterations", default=900000, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)

    # Output
    parser.add_argument("--keep_checkpoints", default=1, type=int)
    parser.add_argument("--checkpoint_period", default=10000, type=int)
    parser.add_argument("--summary_period", default=2000, type=int)

    cl_args = parser.parse_args()

    experiment_path = os.path.join(cl_args.experiments_dir,
                                   cl_args.experiment_name)
    config_path = os.path.join(experiment_path, "conf.ini")

    if os.path.exists(config_path):
        with open(config_path, "r") as fd:
            config_text = fd.read()
        config_args = parser.parse_args(config_file_contents=config_text)

        cl_dict = vars(cl_args)
        config_dict = vars(config_args)

        for k in cl_dict:
            if cl_dict[k] == config_dict[k]:
                continue

            print("Overriding config value {}={} with {}={}".format(
                k, config_dict[k], k, cl_dict[k]))
            config_dict[k] = cl_dict[k]

        args = config_args

    else:
        args = cl_args
        os.makedirs(experiment_path, exist_ok=True)
        parser.write_config_file(args, [config_path])

    return args