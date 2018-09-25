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
        print('Failed to run \"git {}\"'.format(' '.join(args)))

def load_model(repo_url, file_path, callable_name, checkpoint=None, model_dir=None, map_location=None, git_branch='master'):
    r"""
    TODO: doc string
    """
    if model_dir is None:
        torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
        model_dir = os.getenv('TORCH_HUB_DIR', os.path.join(torch_home, 'model'))
        checkpoint_dir = os.getenv('TORCH_HUB_DIR', os.path.join(torch_home, 'checkpoint'))
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # clone the github repo
    git('clone', '-b', git_branch, repo_url, model_dir)
    module_path = os.path.join(model_dir, file_path)
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
        state_dict = model_zoo.load_url(checkpoint, checkpoint_dir, map_location)
        model.load_state_dict(state_dict)

    return model

