import os
import subprocess
from abc import ABC, abstractmethod
from .agent_utils import OllamaLLM, AVAILABLE_OLLAMA_MODELS

# Define a root path for the project if needed, assuming agents are in an 'agents' subdirectory
# This helps in reliably locating the 'shared_library'
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_SHARED_LIBRARY_PATH = os.path.join(PROJECT_ROOT, 'shared_library')

class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    Provides common functionalities like interacting with the shared library
    and utilizing an LLM.
    """
    def __init__(self, llm_model_name: str = None, shared_library_path: str = None):
        if llm_model_name is None:
            llm_model_name = AVAILABLE_OLLAMA_MODELS[0] # Default to the first available model
        self.llm = OllamaLLM(model_name=llm_model_name)

        self.shared_library_path = shared_library_path or DEFAULT_SHARED_LIBRARY_PATH
        if not os.path.isdir(self.shared_library_path):
            print(f"Warning: Shared library path '{self.shared_library_path}' does not exist or is not a directory.")
            # Potentially create it, or handle error appropriately
            # For now, we'll allow it, but operations might fail.
            # os.makedirs(self.shared_library_path, exist_ok=True)


    def list_library_files(self, sub_directory: str = "") -> list[str]:
        """
        Lists files and directories within the shared_library path or a specified subdirectory.
        Args:
            sub_directory (str): Optional. A subdirectory within the shared library.
        Returns:
            list[str]: A list of file and directory names. Returns empty list on error.
        """
        target_path = os.path.join(self.shared_library_path, sub_directory)
        if not os.path.isdir(target_path):
            print(f"Error: Directory '{target_path}' does not exist.")
            return []
        try:
            return os.listdir(target_path)
        except OSError as e:
            print(f"Error listing files in '{target_path}': {e}")
            return []

    def read_file(self, relative_filepath: str) -> str | None:
        """
        Reads the content of a file from the shared library.
        Args:
            relative_filepath (str): The path to the file, relative to the shared_library_path.
        Returns:
            str | None: The content of the file, or None if an error occurs.
        """
        full_path = os.path.join(self.shared_library_path, relative_filepath)
        if not os.path.isfile(full_path):
            print(f"Error: File '{full_path}' does not exist or is not a file.")
            return None
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        except IOError as e:
            print(f"Error reading file '{full_path}': {e}")
            return None

    def propose_code_change(self, relative_filepath: str, new_code_content: str, explanation: str) -> bool:
        """
        Simulates proposing a code change to a file in the shared library.
        In a real system, this would create a diff, request a review, or directly apply if permitted.
        Args:
            relative_filepath (str): The path to the file, relative to the shared_library_path.
            new_code_content (str): The proposed new content for the file.
            explanation (str): An explanation of why this change is proposed.
        Returns:
            bool: True if the proposal is logged (simulated), False otherwise.
        """
        full_path = os.path.join(self.shared_library_path, relative_filepath)
        print(f"--- Code Change Proposal ---")
        print(f"Agent: {self.__class__.__name__} (using LLM: {self.llm.model_name})")
        print(f"File: {full_path}")
        print(f"Explanation: {explanation}")
        print(f"Proposed Content:\n```\n{new_code_content}\n```")
        # In a real system, this would not write directly but go through a change management process.
        # For now, we can simulate writing it for testing purposes if a flag is set,
        # or just log the proposal.
        # For safety in this placeholder, we will NOT write to the file.
        print(f"--- End Proposal ---")
        return True


    def _execute_command(self, command: list[str], working_directory: str = None) -> tuple[bool, str, str]:
        """
        Executes a shell command.
        Args:
            command (list[str]): The command and its arguments as a list.
            working_directory (str, optional): The directory to execute the command in. Defaults to None.
        Returns:
            tuple[bool, str, str]: Success (True/False), stdout (str), stderr (str).
        """
        try:
            process = subprocess.run(command, capture_output=True, text=True, cwd=working_directory, check=False)
            if process.returncode == 0:
                print(f"Command '{' '.join(command)}' executed successfully.")
                return True, process.stdout, process.stderr
            else:
                print(f"Command '{' '.join(command)}' failed with error code {process.returncode}.")
                print(f"Stderr: {process.stderr}")
                return False, process.stdout, process.stderr
        except Exception as e:
            print(f"Exception executing command '{' '.join(command)}': {e}")
            return False, "", str(e)

    def run_code_snippet(self, code_string: str, context_description: str = "") -> dict:
        """
        Simulates running a Python code snippet in a controlled environment.
        In a real implementation, this would involve sandboxing and resource limits.
        Args:
            code_string (str): The Python code to run.
            context_description (str): A description of what this code is trying to achieve (for logging).
        Returns:
            dict: A dictionary containing results, e.g.,
                  {'success': bool, 'output': str, 'error': str, 'metrics': dict}
                  Metrics could include placeholder for time taken, memory used.
        """
        print(f"Agent {self.__class__.__name__}: Simulating run of code snippet for '{context_description}'.")
        print(f"Code:\n```python\n{code_string}\n```")

        # Placeholder for actual execution.
        # In a real system, you'd write this to a temp file, run it with subprocess,
        # capture output, and potentially monitor resources.

        # For now, just simulate success.
        # A more advanced simulation could try to `exec()` it, but that's risky.
        # Or, use ast.parse to check for basic Python syntax.
        try:
            compile(code_string, '<string>', 'exec')
            return {
                "success": True,
                "output": "Simulated execution successful.",
                "error": "",
                "metrics": {"time_taken_ms": 10.0, "memory_used_kb": 50.0} # Dummy metrics
            }
        except SyntaxError as e:
            print(f"Syntax error in provided code snippet: {e}")
            return {
                "success": False,
                "output": "",
                "error": f"SyntaxError: {e}",
                "metrics": {}
            }

    @abstractmethod
    def perform_task(self, task_description: str) -> None:
        """
        An abstract method that subclasses must implement to perform their specific tasks.
        """
        pass

if __name__ == '__main__':
    # Create a dummy shared_library structure for testing
    TEST_SHARED_LIB_PATH = os.path.join(PROJECT_ROOT, 'temp_shared_library_for_test')
    DUMMY_FILE_REL_PATH = "dummy_module.py"
    DUMMY_FILE_ABS_PATH = os.path.join(TEST_SHARED_LIB_PATH, DUMMY_FILE_REL_PATH)
    DUMMY_SUBDIR = "utils"
    DUMMY_SUBDIR_FILE_REL_PATH = os.path.join(DUMMY_SUBDIR, "helpers.py")
    DUMMY_SUBDIR_FILE_ABS_PATH = os.path.join(TEST_SHARED_LIB_PATH, DUMMY_SUBDIR_FILE_REL_PATH)

    os.makedirs(os.path.join(TEST_SHARED_LIB_PATH, DUMMY_SUBDIR), exist_ok=True)
    with open(DUMMY_FILE_ABS_PATH, 'w') as f:
        f.write("def hello_world():\n    print('Hello from dummy module!')")
    with open(DUMMY_SUBDIR_FILE_ABS_PATH, 'w') as f:
        f.write("def utility_func():\n    return 'Utility processed'")

    # Dummy Agent for testing BaseAgent functionalities
    class TestAgent(BaseAgent):
        def perform_task(self, task_description: str) -> None:
            print(f"TestAgent performing task: {task_description}")
            # Example of using LLM
            analysis = self.llm.analyze_code("print('hello')", "Is this code complete?")
            print(f"LLM Analysis: {analysis}")

    print(f"Using shared library path: {TEST_SHARED_LIB_PATH}")
    agent = TestAgent(shared_library_path=TEST_SHARED_LIB_PATH)

    print("\nListing root of shared library:")
    files = agent.list_library_files()
    print(files)
    assert DUMMY_FILE_REL_PATH in files
    assert DUMMY_SUBDIR in files

    print(f"\nListing '{DUMMY_SUBDIR}' subdirectory:")
    subdir_files = agent.list_library_files(DUMMY_SUBDIR)
    print(subdir_files)
    assert "helpers.py" in subdir_files

    print(f"\nReading '{DUMMY_FILE_REL_PATH}':")
    content = agent.read_file(DUMMY_FILE_REL_PATH)
    print(content)
    assert content == "def hello_world():\n    print('Hello from dummy module!')"

    print("\nProposing a code change (simulation):")
    agent.propose_code_change(DUMMY_FILE_REL_PATH, "def new_hello():\n    print('New hello!')", "Improved greeting")

    print("\nRunning a code snippet (simulation):")
    result = agent.run_code_snippet("x = 1 + 1\nprint(x)", "Test addition")
    print(result)
    assert result["success"]

    print("\nPerforming a task (TestAgent specific):")
    agent.perform_task("Demonstrate LLM usage.")

    # Clean up dummy shared_library
    import shutil
    shutil.rmtree(TEST_SHARED_LIB_PATH)
    print(f"\nCleaned up test directory: {TEST_SHARED_LIB_PATH}")

    print("\nBaseAgent tests completed.")
