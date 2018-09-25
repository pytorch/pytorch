import torch
import os
import importlib
import torch.utils.model_zoo as model_zoo
import shutil

import subprocess


def git(*args):
    FNULL = open(os.devnull, 'w')
    try:
        subprocess.check_call(['git'] + list(args), stdout=FNULL, stderr=FNULL)
    except subprocess.CalledProcessError as err:
        print('Failed to run \"git {}\"\nAborting...'.format(' '.join(args)))
        exit(-1)


def load_model(
        file_path,
        callable_name,
        checkpoint=None,
        hub_dir=None,
        map_location=None,
        git_repo=None,
        git_branch='master'):
    r"""
    Load a model from local/github, with pretrained weights

    Args:
        file_path: required, can be a relative path to file, or a full path if file is local.
        callable_name: required, a funciton in :attr"`file_path` that returns the model
        checkpoint: URL to download the checkpoint for the model
        hub_dir: local path to save the intermediate model & checkpoint files
        map_location: destination device when loading checkpoint
        git_repo: URL to github repo where the model is saved
        git_branch: branch of the github repo

    Returns:
        model: a model with pretrained weights( if :attr:`checkpoint` is specified)
    """
    if hub_dir is None:
        hub_dir = os.path.expanduser(os.getenv('TORCH_HUB_DIR', '~/.torch/hub'))
    if git_repo:
        model_dir = os.path.join(hub_dir, 'model')
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)

        # clone the github repo
        git('clone', '-b', git_branch, git_repo, model_dir)
        module_path = os.path.join(model_dir, file_path)
    else:
        module_path = file_path

    try:
        # importlib.util was added since python3.4
        spec = importlib.util.spec_from_file_location(file_path, module_path)
        loaded_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded_module)
    except AttributeError:
        import imp
        loaded_module = imp.load_source(file_path, module_path)

    if callable_name not in dir(loaded_module):
        raise RuntimeError('callable {} doesn\'t exist in {}'.format(callable_name, loaded_module))

    func = getattr(loaded_module, callable_name)
    if not callable(func):
        raise RuntimeError('{} is not callable'.format(callable_name))
    model = func()

    if checkpoint is None:
        print('No pretrained weights provided. Proceeding with random initialized weights...')
    else:
        checkpoint_dir = os.path.join(hub_dir, 'checkpoint')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        model.load_state_dict(model_zoo.load_url(checkpoint, checkpoint_dir, map_location))

    return model
