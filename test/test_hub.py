import unittest
from urllib.error import URLError
import os
import http
import tempfile
import shutil

import torch
import torch.hub as hub
from torch.testing._internal.common_utils import retry, IS_SANDCASTLE
from torch.testing._internal.common_utils import TestCase


def sum_of_state_dict(state_dict):
    s = 0
    for _, v in state_dict.items():
        s += v.sum()
    return s

SUM_OF_HUB_EXAMPLE = 431080
TORCHHUB_EXAMPLE_RELEASE_URL = 'https://github.com/ailzhang/torchhub_example/releases/download/0.1/mnist_init_ones'


@unittest.skipIf(IS_SANDCASTLE, 'Sandcastle cannot ping external')
class TestHub(TestCase):
    @retry(URLError, tries=3)
    def test_load_from_github(self):
        hub_model = hub.load(
            'ailzhang/torchhub_example',
            'mnist',
            source='github',
            pretrained=True,
            verbose=False)
        self.assertEqual(sum_of_state_dict(hub_model.state_dict()),
                         SUM_OF_HUB_EXAMPLE)

    @retry(URLError, tries=3)
    def test_load_from_local_dir(self):
        local_dir = hub._get_cache_or_reload(
            'ailzhang/torchhub_example', force_reload=False)
        hub_model = hub.load(
            local_dir,
            'mnist',
            source='local',
            pretrained=True,
            verbose=False)
        self.assertEqual(sum_of_state_dict(hub_model.state_dict()),
                         SUM_OF_HUB_EXAMPLE)

    @retry(URLError, tries=3)
    def test_load_from_branch(self):
        hub_model = hub.load(
            'ailzhang/torchhub_example:ci/test_slash',
            'mnist',
            pretrained=True,
            verbose=False)
        self.assertEqual(sum_of_state_dict(hub_model.state_dict()),
                         SUM_OF_HUB_EXAMPLE)

    @retry(URLError, tries=3)
    def test_set_dir(self):
        temp_dir = tempfile.gettempdir()
        hub.set_dir(temp_dir)
        hub_model = hub.load(
            'ailzhang/torchhub_example',
            'mnist',
            pretrained=True,
            verbose=False)
        self.assertEqual(sum_of_state_dict(hub_model.state_dict()),
                         SUM_OF_HUB_EXAMPLE)
        assert os.path.exists(temp_dir + '/ailzhang_torchhub_example_master')
        shutil.rmtree(temp_dir + '/ailzhang_torchhub_example_master')

    @retry(URLError, tries=3)
    def test_list_entrypoints(self):
        entry_lists = hub.list('ailzhang/torchhub_example', force_reload=True)
        self.assertObjectIn('mnist', entry_lists)

    @retry(URLError, tries=3)
    def test_download_url_to_file(self):
        temp_file = os.path.join(tempfile.gettempdir(), 'temp')
        hub.download_url_to_file(TORCHHUB_EXAMPLE_RELEASE_URL, temp_file, progress=False)
        loaded_state = torch.load(temp_file)
        self.assertEqual(sum_of_state_dict(loaded_state),
                         SUM_OF_HUB_EXAMPLE)

    @retry(URLError, tries=3)
    @retry(http.client.RemoteDisconnected, tries=3)
    def test_load_state_dict_from_url(self):
        loaded_state = hub.load_state_dict_from_url(TORCHHUB_EXAMPLE_RELEASE_URL)
        self.assertEqual(sum_of_state_dict(loaded_state),
                         SUM_OF_HUB_EXAMPLE)

    @retry(URLError, tries=3)
    def test_load_zip_checkpoint(self):
        hub_model = hub.load(
            'ailzhang/torchhub_example',
            'mnist_zip',
            pretrained=True,
            verbose=False)
        self.assertEqual(sum_of_state_dict(hub_model.state_dict()),
                         SUM_OF_HUB_EXAMPLE)

    # Test the default zipfile serialization format produced by >=1.6 release.
    @retry(URLError, tries=3)
    def test_load_zip_1_6_checkpoint(self):
        hub_model = hub.load(
            'ailzhang/torchhub_example',
            'mnist_zip_1_6',
            pretrained=True,
            verbose=False)
        self.assertEqual(sum_of_state_dict(hub_model.state_dict()),
                         SUM_OF_HUB_EXAMPLE)


    def test_hub_dir(self):
        with tempfile.TemporaryDirectory('hub_dir') as dirname:
            torch.hub.set_dir(dirname)
            self.assertEqual(torch.hub.get_dir(), dirname)

    @retry(URLError, tries=3)
    def test_hub_parse_repo_info(self):
        # If the branch is specified we just parse the input and return
        self.assertEqual(
            torch.hub._parse_repo_info('a/b:c'),
            ('a', 'b', 'c')
        )
        # For torchvision, the default branch is main
        self.assertEqual(
            torch.hub._parse_repo_info('pytorch/vision'),
            ('pytorch', 'vision', 'main')
        )
        # For the torchhub_example repo, the default branch is still master
        self.assertEqual(
            torch.hub._parse_repo_info('ailzhang/torchhub_example'),
            ('ailzhang', 'torchhub_example', 'master')
        )

    @retry(URLError, tries=3)
    def test_load_state_dict_from_url_with_name(self):
        with tempfile.TemporaryDirectory('hub_dir') as dirname:
            torch.hub.set_dir(dirname)
            file_name = 'test_file'
            loaded_state = hub.load_state_dict_from_url(TORCHHUB_EXAMPLE_RELEASE_URL, file_name=file_name)
            self.assertTrue(os.path.exists(os.path.join(dirname, 'checkpoints', file_name)))
            self.assertEqual(sum_of_state_dict(loaded_state),
                             SUM_OF_HUB_EXAMPLE)

    @retry(URLError, tries=3)
    def test_load_commit_from_forked_repo(self):
        with self.assertRaisesRegex(
                ValueError,
                'If it\'s a commit from a forked repo'):
            model = torch.hub.load('pytorch/vision:4e2c216', 'resnet18', force_reload=True)
