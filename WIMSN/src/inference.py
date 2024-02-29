import sys
from types import SimpleNamespace as SN
from collections.abc import Mapping
from copy import deepcopy
import numpy as np
import torch
import yaml
import copy
import pdb
import os
import wandb
import warnings
# warnings.filterwarnings("ignore")

from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from controllers import REGISTRY as mac_REGISTRY
from runners import REGISTRY as r_REGISTRY
from run import REGISTRY as run_REGISTRY
from utils.logging import get_logger


def args_sanity_check(config):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not torch.cuda.is_available():
        config["use_cuda"] = False

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"] // config["batch_size_run"]
        ) * config["batch_size_run"]

    return config

def _get_config(params, arg_name, subfolder):
    
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(
            os.path.join(
                os.path.dirname(__file__),
                "config",
                subfolder,
                "{}.yaml".format(config_name),
            ),
            "r",
        ) as f:
            try:
                config_dict = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        config_dict.update({arg_name: config_name})
        return config_dict
    else:
        return {}

def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


if __name__ == "__main__":

    params = deepcopy(sys.argv)

    # Load algorithm and env base configs
    config_dict = {}
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    _config = args_sanity_check(config_dict)
    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"
    args.env_args.update({'seed': 101})

    runner = r_REGISTRY['evaluate'](args=args)
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.accumulated_episodes = getattr(args, "accumulated_episodes", None)

    model_paths = [
        'results/ippo_ir_1000_CapacityHigh_seed101_0406T2124/models/4381600',
        'results/ippo_ir_1000_Standard_seed101_0404T0934/models/4381600',
        'results/ippo_ir_1000_CapacityLow_seed101_0407T1714/models/4381600',
        'results/ippo_ir_1000_CapacityLowest_seed101_0409T0843/models/4381600',
    ]

    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "mean_action": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": torch.float,
        },
        "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
        "avail_actions": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": torch.int,
        },
        "probs": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": torch.float,
        },
        "reward": {"vshape": (1,)},
        "individual_rewards": {"vshape": (1,), "group": "agents"},
        "cur_balance": {"vshape": (1,), "group": "agents"},
        "terminated": {"vshape": (1,), "dtype": torch.uint8},
    }
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}
    buffer = ReplayBuffer(
        scheme,
        groups,
        args.buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )

    controllers = []
    for path in model_paths:
        controllers.append(mac_REGISTRY['mappo_mac'](buffer.scheme, groups, args))
        controllers[-1].load_models(path)
        controllers[-1].cuda()

    test_args = copy.deepcopy(args)
    test_args.env_args["mode"] = "test"

    test_runner = r_REGISTRY['evaluate'](args=test_args)
    test_runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=controllers[1])
    record_rewards = \
        test_runner.run(test_mode=True, lbda_index=0, storage_capacity=[50000, 100000])
    np.save('results/envinc_standard', record_rewards)

    test_runner = r_REGISTRY['evaluate'](args=test_args)
    test_runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=controllers[2])
    record_rewards = \
        test_runner.run(test_mode=True, lbda_index=0, storage_capacity=[50000, 100000])
    np.save('results/envinc_capacitylow', record_rewards)

    test_runner = r_REGISTRY['evaluate'](args=test_args)
    test_runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=controllers[2])
    record_rewards = \
        test_runner.run(test_mode=True, lbda_index=0, storage_capacity=[100000, 100000])
    np.save('results/envstd_capacitylow', record_rewards)

    test_runner = r_REGISTRY['evaluate'](args=test_args)
    test_runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=controllers[2])
    record_rewards = \
        test_runner.run(test_mode=True, lbda_index=0, storage_capacity=[50000, 50000])
    np.save('results/envlow_capacitylow', record_rewards)

    test_runner = r_REGISTRY['evaluate'](args=test_args)
    test_runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=controllers[1])
    record_rewards = \
        test_runner.run(test_mode=True, lbda_index=0, storage_capacity=[100000, 100000])
    np.save('results/envstd_standard', record_rewards)

    test_runner = r_REGISTRY['evaluate'](args=test_args)
    test_runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=controllers[1])
    record_rewards = \
        test_runner.run(test_mode=True, lbda_index=0, storage_capacity=[50000, 50000])
    np.save('results/envlow_standard', record_rewards)

    test_runner = r_REGISTRY['evaluate'](args=test_args)
    test_runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=controllers[1])
    record_rewards = \
        test_runner.run(test_mode=True, lbda_index=0, storage_capacity=[100000, 50000])
    np.save('results/envdec_standard', record_rewards)

    test_runner = r_REGISTRY['evaluate'](args=test_args)
    test_runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=controllers[2])
    record_rewards = \
        test_runner.run(test_mode=True, lbda_index=0, storage_capacity=[100000, 50000])
    np.save('results/envdec_capacitylow', record_rewards)
