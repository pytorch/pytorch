# Owner(s): ["module: hub"]

import unittest
import os
import tempfile
import warnings

import torch
import torch.hub as hub
from torch.testing._internal.common_utils import retry, IS_SANDCASTLE, TestCase


def sum_of_state_dict(state_dict):
    s = 0
    for _, v in state_dict.items():
        s += v.sum()
    return s


SUM_OF_HUB_EXAMPLE = 431080
TORCHHUB_EXAMPLE_RELEASE_URL = 'https://github.com/ailzhang/torchhub_example/releases/download/0.1/mnist_init_ones'


@unittest.skipIf(IS_SANDCASTLE, 'Sandcastle cannot ping external')
class TestHub(TestCase):

    def setUp(self):
        super().setUp()
        self.previous_hub_dir = torch.hub.get_dir()
        self.tmpdir = tempfile.TemporaryDirectory('hub_dir')
        torch.hub.set_dir(self.tmpdir.name)

    def tearDown(self):
        super().tearDown()
        torch.hub.set_dir(self.previous_hub_dir)  # probably not needed, but can't hurt
        self.tmpdir.cleanup()

    @retry(Exception, tries=3)
    def test_load_from_github(self):
        hub_model = hub.load('ailzhang/torchhub_example', 'mnist', source='github', pretrained=True, verbose=False)
        self.assertEqual(sum_of_state_dict(hub_model.state_dict()), SUM_OF_HUB_EXAMPLE)

    @retry(Exception, tries=3)
    def test_load_from_local_dir(self):
        local_dir = hub._get_cache_or_reload('ailzhang/torchhub_example', force_reload=False)
        hub_model = hub.load(local_dir, 'mnist', source='local', pretrained=True, verbose=False)
        self.assertEqual(sum_of_state_dict(hub_model.state_dict()), SUM_OF_HUB_EXAMPLE)

    @retry(Exception, tries=3)
    def test_load_from_branch(self):
        hub_model = hub.load('ailzhang/torchhub_example:ci/test_slash', 'mnist', pretrained=True, verbose=False)
        self.assertEqual(sum_of_state_dict(hub_model.state_dict()), SUM_OF_HUB_EXAMPLE)

    @retry(Exception, tries=3)
    def test_get_set_dir(self):
        previous_hub_dir = torch.hub.get_dir()
        with tempfile.TemporaryDirectory('hub_dir') as tmpdir:
            torch.hub.set_dir(tmpdir)
            self.assertEqual(torch.hub.get_dir(), tmpdir)
            self.assertNotEqual(previous_hub_dir, tmpdir)

            hub_model = hub.load('ailzhang/torchhub_example', 'mnist', pretrained=True, verbose=False)
            self.assertEqual(sum_of_state_dict(hub_model.state_dict()), SUM_OF_HUB_EXAMPLE)
            assert os.path.exists(os.path.join(tmpdir, 'ailzhang_torchhub_example_master'))

        # Test that set_dir properly calls expanduser()
        # non-regression test for https://github.com/pytorch/pytorch/issues/69761
        new_dir = os.path.join("~", "hub")
        torch.hub.set_dir(new_dir)
        self.assertEqual(torch.hub.get_dir(), os.path.expanduser(new_dir))


    @retry(Exception, tries=3)
    def test_list_entrypoints(self):
        entry_lists = hub.list('ailzhang/torchhub_example', force_reload=True)
        self.assertObjectIn('mnist', entry_lists)

    @retry(Exception, tries=3)
    def test_download_url_to_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = os.path.join(tmpdir, 'temp')
            hub.download_url_to_file(TORCHHUB_EXAMPLE_RELEASE_URL, f, progress=False)
            loaded_state = torch.load(f)
            self.assertEqual(sum_of_state_dict(loaded_state), SUM_OF_HUB_EXAMPLE)

    @retry(Exception, tries=3)
    def test_load_state_dict_from_url(self):
        loaded_state = hub.load_state_dict_from_url(TORCHHUB_EXAMPLE_RELEASE_URL)
        self.assertEqual(sum_of_state_dict(loaded_state), SUM_OF_HUB_EXAMPLE)

        # with name
        file_name = "the_file_name"
        loaded_state = hub.load_state_dict_from_url(TORCHHUB_EXAMPLE_RELEASE_URL, file_name=file_name)
        expected_file_path = os.path.join(torch.hub.get_dir(), 'checkpoints', file_name)
        self.assertTrue(os.path.exists(expected_file_path))
        self.assertEqual(sum_of_state_dict(loaded_state), SUM_OF_HUB_EXAMPLE)

    @retry(Exception, tries=3)
    def test_load_legacy_zip_checkpoint(self):
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            hub_model = hub.load('ailzhang/torchhub_example', 'mnist_zip', pretrained=True, verbose=False)
            self.assertEqual(sum_of_state_dict(hub_model.state_dict()), SUM_OF_HUB_EXAMPLE)
            assert any("will be deprecated in favor of default zipfile" in str(w) for w in ws)

    # Test the default zipfile serialization format produced by >=1.6 release.
    @retry(Exception, tries=3)
    def test_load_zip_1_6_checkpoint(self):
        hub_model = hub.load('ailzhang/torchhub_example', 'mnist_zip_1_6', pretrained=True, verbose=False)
        self.assertEqual(sum_of_state_dict(hub_model.state_dict()), SUM_OF_HUB_EXAMPLE)

    @retry(Exception, tries=3)
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

    @retry(Exception, tries=3)
    def test_load_commit_from_forked_repo(self):
        with self.assertRaisesRegex(ValueError, 'If it\'s a commit from a forked repo'):
            torch.hub.load('pytorch/vision:4e2c216', 'resnet18', force_reload=True)