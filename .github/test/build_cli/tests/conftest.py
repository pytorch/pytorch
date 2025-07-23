import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

# Add the parent directory to sys.path to allow importing the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def mock_subprocess_run():
    """Mock subprocess.run to avoid executing actual commands during tests"""
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = MagicMock()
        yield mock_run

@pytest.fixture
def mock_os_environ():
    """Mock os.environ to control environment variables during tests"""
    with patch.dict('os.environ', {}, clear=True) as mock_env:
        yield mock_env

@pytest.fixture
def mock_path_exists():
    """Mock Path.exists to control file existence checks"""
    with patch('pathlib.Path.exists') as mock_exists:
        mock_exists.return_value = True
        yield mock_exists

@pytest.fixture
def mock_path_read_text():
    """Mock Path.read_text to control file content reading"""
    with patch('pathlib.Path.read_text') as mock_read_text:
        mock_read_text.return_value = "test-commit-hash"
        yield mock_read_text

@pytest.fixture
def mock_os_makedirs():
    """Mock os.makedirs to avoid creating actual directories"""
    with patch('os.makedirs') as mock_makedirs:
        yield mock_makedirs

@pytest.fixture
def mock_shutil_rmtree():
    """Mock shutil.rmtree to avoid removing actual directories"""
    with patch('shutil.rmtree') as mock_rmtree:
        yield mock_rmtree

@pytest.fixture
def mock_os_path_exists():
    """Mock os.path.exists to control path existence checks"""
    with patch('os.path.exists') as mock_exists:
        mock_exists.return_value = True
        yield mock_exists

@pytest.fixture
def mock_os_path_abspath():
    def side_effect(path):
        return f"/mocked/path/{path}"
        
    with patch('os.path.abspath') as mock_abspath:
        mock_abspath.side_effect = side_effect
        yield mock_abspath
@pytest.fixture
def mock_timer():
    """Mock the Timer context manager"""
    with patch('utils.Timer.__enter__') as mock_enter, \
         patch('utils.Timer.__exit__') as mock_exit:
        mock_enter.return_value = None
        mock_exit.return_value = None
        yield (mock_enter, mock_exit)
