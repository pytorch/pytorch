import os
import random
from enum import Enum, auto

from ..base_agent import BaseAgent, DEFAULT_SHARED_LIBRARY_PATH
from ..agent_utils import AVAILABLE_OLLAMA_MODELS

class ImprovementStrategy(Enum):
    GENERAL_ANALYSIS = auto()
    LINE_BY_LINE = auto()
    SECTION = auto()
    NEW_PARADIGM_CUSTOM_DATA_TYPE = auto()
    NEW_PARADIGM_LAYER_DESIGN = auto() # Placeholder for now

class FrameworkImproveAgent(BaseAgent):
    """
    An agent responsible for analyzing and improving the shared codebase using various strategies.
    """
    def __init__(self, llm_model_name: str = None, shared_library_path: str = None):
        super().__init__(llm_model_name=llm_model_name, shared_library_path=shared_library_path)
        # print(f"FrameworkImproveAgent initialized, using LLM: {self.llm.model_name}, Library: {self.shared_library_path}") # Less verbose

    def _general_analysis_improvement(self, relative_filepath: str):
        """
        Performs a general analysis of the file to identify and propose an improvement.
        """
        print(f"\nFrameworkImproveAgent (General Analysis): Identifying improvement for '{relative_filepath}'...")
        file_content = self.read_file(relative_filepath)
        if file_content is None:
            print(f"FrameworkImproveAgent (General Analysis): Could not read file {relative_filepath}. Aborting.")
            return

        analysis_prompt = (
            "Analyze the following Python code. "
            "Identify one specific area for improvement (e.g., performance, readability, error handling, refactoring for clarity). "
            "Briefly describe the current state and the suggested improvement. "
            "Be specific and actionable. If no obvious improvement, state 'No specific improvement identified'."
        )
        analysis_result = self.llm.analyze_code(code_content=file_content, analysis_prompt=analysis_prompt)
        # print(f"LLM Analysis (General) for '{relative_filepath}':\n{analysis_result}") # Can be verbose

        if "No specific improvement identified" in analysis_result or "No obvious improvement" in analysis_result :
            print("FrameworkImproveAgent (General Analysis): LLM found no specific improvement.")
            return

        generation_prompt = (
            f"Based on the analysis that suggested an improvement for the code below, "
            f"please generate the complete, improved version of the following Python code snippet. "
            f"Original code:\n```python\n{file_content}\n```\n"
            f"Analysis and suggestion: {analysis_result}\n"
            f"Generate only the full improved Python code. Ensure it's a direct replacement or enhancement."
        )

        improved_code_suggestion = self.llm.generate_code(prompt=generation_prompt)
        # print(f"LLM Suggested Improved Code (General) for '{relative_filepath}':\n{improved_code_suggestion}")

        benchmark_passed = self.simulate_benchmark(original_code=file_content, improved_code=improved_code_suggestion, context=f"General Analysis: {relative_filepath}")

        if benchmark_passed:
            explanation = (
                f"LLM suggested an improvement for {relative_filepath} via general analysis: {analysis_result}. "
                "The proposed code passed simulated benchmark."
            )
            self.propose_code_change(relative_filepath, improved_code_suggestion, explanation)
        else:
            print(f"FrameworkImproveAgent (General Analysis): Simulated benchmark failed for {relative_filepath}. Not proposing.")

    def _improve_line_by_line(self, relative_filepath: str, file_content: str):
        print(f"\nFrameworkImproveAgent (Line-by-Line): Attempting for '{relative_filepath}'")
        lines = file_content.splitlines()
        if not lines:
            print("FrameworkImproveAgent (Line-by-Line): File is empty.")
            return

        num_lines_to_try = min(len(lines), 3) # Try to improve up to 3 random lines
        line_indices_to_try = random.sample(range(len(lines)), num_lines_to_try)

        improved_something_successfully = False
        for line_idx in line_indices_to_try:
            original_line = lines[line_idx]
            if not original_line.strip() or original_line.strip().startswith("#"): # Skip empty or comment lines
                print(f"FrameworkImproveAgent (Line-by-Line): Skipping blank or comment line {line_idx+1}: '{original_line}'")
                continue

            print(f"FrameworkImproveAgent (Line-by-Line): Analyzing line {line_idx+1}: '{original_line}'")

            # For very short lines, LLM might not find much or overthink.
            # We can add a simple heuristic, e.g. if line is just "pass" or "return x"
            if len(original_line.strip()) < 5 and not any(c in original_line for c in "=+*/%&|<>[]{}()"):
                 print(f"FrameworkImproveAgent (Line-by-Line): Skipping very simple line {line_idx+1}: '{original_line}'")
                 continue

            line_analysis_prompt = (
                f"Analyze *only* the following single line of Python code: ```python\n{original_line}\n```"
                f"Suggest a more optimal, robust, or idiomatic version of *just this line*. "
                f"If no improvement is clear or the line is trivial (e.g., a simple print, a comment, basic assignment without calculation), "
                f"respond *only* with the exact phrase 'No change needed for this line.' "
                f"Otherwise, provide a brief explanation of the problem and then the suggested improved line itself prefixed by 'Improved line: '."
            )
            analysis_result = self.llm.analyze_code(code_content=original_line, analysis_prompt=line_analysis_prompt)
            # print(f"Line Analysis Result: {analysis_result}") # Verbose

            if "No change needed for this line." in analysis_result:
                print(f"FrameworkImproveAgent (Line-by-Line): LLM suggested no change for line {line_idx+1}.")
                continue

            # Try to extract the improved line suggestion if analysis_result contains it
            improved_line_candidate = None
            if "Improved line: " in analysis_result:
                improved_line_candidate = analysis_result.split("Improved line: ", 1)[-1].strip()

            if not improved_line_candidate: # If LLM didn't follow format, try to generate
                line_generation_prompt = (
                    f"Original Python line: ```python\n{original_line}\n```"
                    f"Based on analysis: '{analysis_result}', provide *only* the single, improved line of Python code. "
                    f"Do not include any explanation or markdown."
                )
                improved_line_candidate = self.llm.generate_code(prompt=line_generation_prompt).strip()

            if not improved_line_candidate or improved_line_candidate == original_line:
                print(f"FrameworkImproveAgent (Line-by-Line): No effective change generated for line {line_idx+1}.")
                continue

            # Ensure it's really a single line (LLMs can sometimes return more)
            improved_line_candidate = improved_line_candidate.splitlines()[0]

            print(f"FrameworkImproveAgent (Line-by-Line): Original line {line_idx+1}: '{original_line}'")
            print(f"FrameworkImproveAgent (Line-by-Line): Suggested new line: '{improved_line_candidate}'")

            # Create new file content with this single line changed
            temp_lines = list(lines) # Make a copy
            temp_lines[line_idx] = improved_line_candidate
            modified_full_content = "\n".join(temp_lines)

            # Check if the change is trivial (e.g. just whitespace)
            if original_line.strip() == improved_line_candidate.strip():
                print(f"FrameworkImproveAgent (Line-by-Line): Change for line {line_idx+1} is trivial (whitespace). Skipping.")
                continue

            benchmark_context = f"Line-by-Line: {relative_filepath}, line {line_idx+1}"
            if self.simulate_benchmark(original_code=file_content, improved_code=modified_full_content, context=benchmark_context):
                explanation = (
                    f"LLM suggested an improvement for line {line_idx+1} in {relative_filepath}.\n"
                    f"Original line: '{original_line}'\n"
                    f"Improved line: '{improved_line_candidate}'\n"
                    f"Analysis: {analysis_result.split('Improved line:')[0].strip()}"
                )
                self.propose_code_change(relative_filepath, modified_full_content, explanation)
                improved_something_successfully = True
                print(f"FrameworkImproveAgent (Line-by-Line): Successfully proposed change for line {line_idx+1}. Stopping line-by-line for this file pass.")
                break # Stop after one successful improvement for this file pass
            else:
                print(f"FrameworkImproveAgent (Line-by-Line): Simulated benchmark failed for change in line {line_idx+1}.")

        if not improved_something_successfully:
            print(f"FrameworkImproveAgent (Line-by-Line): No lines were successfully improved and benchmarked in {relative_filepath} this pass.")


    def _improve_section_by_section(self, relative_filepath: str, file_content: str):
        print(f"\nFrameworkImproveAgent (Section): Placeholder for '{relative_filepath}'")
        # Future: identify sections (e.g. functions, or use LLM to suggest a block)
        # For now, let's try to treat the whole file as one section for simplicity to test flow
        if len(file_content.splitlines()) < 5 : # Arbitrary: don't bother for very small files with section analysis
             print(f"FrameworkImproveAgent (Section): File '{relative_filepath}' too short for section analysis. Skipping.")
             return

        print(f"FrameworkImproveAgent (Section): Analyzing entire file '{relative_filepath}' as one section.")
        section_analysis_prompt = (
            f"Analyze the following Python code block (entire file content):\n```python\n{file_content}\n```\n"
            f"Identify a significant section (e.g., a whole function, a complex loop, or a class method) that could be refactored for "
            f"better performance, clarity, or robustness. Describe the original section and your suggested improved version of that section. "
            f"If no single section stands out for improvement, state 'No specific section improvement identified'. "
            f"If an improvement is suggested, provide the complete code for the *improved section only*."
        )
        analysis_and_improved_section = self.llm.analyze_code(code_content=file_content, analysis_prompt=section_analysis_prompt)

        if "No specific section improvement identified" in analysis_and_improved_section:
            print(f"FrameworkImproveAgent (Section): LLM found no specific section improvement for '{relative_filepath}'.")
            return

        # This is tricky: LLM must return identifiable original section and the new section.
        # For simulation, we'll assume LLM provides the *full improved file content* if it makes a section change.
        # This simplifies not having to splice the section back in.
        # A more robust solution would require complex parsing of LLM output.
        print(f"FrameworkImproveAgent (Section): LLM suggested section improvement (details below). Assuming it provided full improved file.")
        # print(f"LLM Analysis/Suggestion (Section) for '{relative_filepath}':\n{analysis_and_improved_section}") # Can be verbose

        # Let's assume the LLM returns the *entire file* with the section improved.
        # This is a simplification for the simulation.
        improved_full_content = analysis_and_improved_section # Placeholder for actual section extraction and replacement

        # Crude check: did it return something that looks like code?
        if "def " not in improved_full_content and "class " not in improved_full_content and "import " not in improved_full_content:
            print(f"FrameworkImproveAgent (Section): LLM output for section improvement doesn't look like full code. Might be just analysis.")
            # Potentially try to generate the full code if only a section was returned + analysis
            # generation_prompt = f"""Original code:
# ```python
# {file_content}
# ```
# LLM suggested this improved section and analysis:
# {analysis_and_improved_section}
#
# Please provide the *entire file content* with this section improved."""
            # improved_full_content = self.llm.generate_code(prompt=generation_prompt)

        if improved_full_content.strip() == file_content.strip():
            print(f"FrameworkImproveAgent (Section): Suggested improvement for '{relative_filepath}' resulted in no change to file content.")
            return

        benchmark_context = f"Section: {relative_filepath}"
        if self.simulate_benchmark(original_code=file_content, improved_code=improved_full_content, context=benchmark_context):
            explanation = (
                f"LLM suggested a section-based improvement for {relative_filepath}. "
                f"Details: {analysis_and_improved_section[:200]}... " # Show start of LLM reasoning
                "The proposed code (assumed to be full file) passed simulated benchmark."
            )
            self.propose_code_change(relative_filepath, improved_full_content, explanation)
            print(f"FrameworkImproveAgent (Section): Successfully proposed section-based change for {relative_filepath}.")
        else:
            print(f"FrameworkImproveAgent (Section): Simulated benchmark failed for section-based change in {relative_filepath}.")


    def _propose_new_custom_data_type(self, relative_filepath: str, file_content: str):
        print(f"\nFrameworkImproveAgent (New Custom Data Type): Analyzing '{relative_filepath if relative_filepath else 'library'}'")

        context_code = file_content if file_content and file_content != "--No specific file content--" else "the entire shared library"

        prompt = (
            f"Consider the codebase context: {context_code}. "
            f"Could a *novel custom data type* (e.g., a new class for specialized tensors, complex parameters, graph nodes, or a unique data structure) "
            f"significantly enhance functionality, efficiency, or clarity if added to the shared library? "
            f"If yes, describe the proposed data type, its purpose, key attributes, and methods. "
            f"Then, provide the Python code for this new custom data type class. "
            f"If not applicable or no clear benefit, state 'No new data type needed'."
        )

        suggestion_and_code = self.llm.generate_code(prompt) # Using generate_code as it might be a mix of analysis and code block
        # print(f"LLM Suggestion (Custom Data Type):\n{suggestion_and_code}") # Verbose

        if "No new data type needed" in suggestion_and_code:
            print("FrameworkImproveAgent (New Custom Data Type): LLM suggested no new data type is needed.")
            return

        # Attempt to extract code part (heuristic: look for 'class Xyz:')
        # This is very basic and would need more robust parsing in reality
        code_parts = []
        in_code_block = False
        for line in suggestion_and_code.splitlines():
            if line.strip().startswith("class "):
                in_code_block = True
            if in_code_block:
                code_parts.append(line)
            # Heuristic to end block, e.g. if LLM starts explaining again after code
            if in_code_block and not line.strip() and len(code_parts) > 5: # Empty line after substantial code
                 # Or if it starts with "This class..." etc.
                if len(suggestion_and_code.splitlines()) > (suggestion_and_code.splitlines().index(line) + 1) and \
                   suggestion_and_code.splitlines()[suggestion_and_code.splitlines().index(line)+1].strip().lower().startswith(("this class", "the class", "this data type")):
                    break


        new_data_type_code = "\n".join(code_parts).strip()

        if not new_data_type_code or not new_data_type_code.startswith("class "):
            print("FrameworkImproveAgent (New Custom Data Type): LLM did not provide a clear class definition for the new data type.")
            return

        print(f"FrameworkImproveAgent (New Custom Data Type): LLM proposed new data type code:\n```python\n{new_data_type_code}\n```")

        # For new paradigms, the "benchmark" is more about utility and less about direct replacement.
        # We'll simulate a "review" that passes if the code is syntactically valid.
        # The proposal would be to ADD this code, likely to a new file or a utility module.

        # For now, propose adding it to a new file in shared_library/custom_types/
        # We need to extract the class name for the filename
        class_name_line = new_data_type_code.splitlines()[0] # class ClassName(Base):
        class_name = class_name_line.split("class ")[1].split("(")[0].split(":")[0].strip()

        if not class_name or not class_name.replace("_","").isalnum():
            class_name = "proposed_custom_type" # Fallback
            print(f"FrameworkImproveAgent (New Custom Data Type): Could not reliably extract class name. Using fallback: {class_name}")


        # Benchmark here is just syntax check for the new code itself
        if self.simulate_benchmark(original_code="", improved_code=new_data_type_code, context=f"New Custom Data Type: {class_name}"):
            # Decide where to propose adding this. For now, a new file.
            # This part of the agent's capability (deciding *where* new code goes) is important.
            new_file_relative_path = os.path.join("custom_types", f"{class_name.lower()}.py")

            explanation = (
                f"LLM proposed a new custom data type: {class_name}. "
                f"Purpose and description (from LLM): {suggestion_and_code.split('class ')[0].strip()}\n"
                f"Proposed to be added as new file: {new_file_relative_path}"
            )
            # The propose_code_change is for existing files. We need a way to propose NEW files.
            # For now, we'll use propose_code_change with an empty original content,
            # implying a new file creation if the tooling supports it.
            # Or, we can just print the proposal for now.
            print(f"--- New Code Proposal (New File) ---")
            print(f"Agent: {self.__class__.__name__} (using LLM: {self.llm.model_name})")
            print(f"Proposed New File: {os.path.join(self.shared_library_path, new_file_relative_path)}")
            print(f"Explanation: {explanation}")
            print(f"Proposed Content:\n```python\n{new_data_type_code}\n```")
            print(f"--- End Proposal ---")
            # In a real system, this would trigger a different workflow for adding new modules/files.
            # self.propose_code_change(new_file_relative_path, new_data_type_code, explanation) # If it can create new files
            print(f"FrameworkImproveAgent (New Custom Data Type): Successfully proposed new data type '{class_name}'. (Simulated new file proposal)")
        else:
            print(f"FrameworkImproveAgent (New Custom Data Type): Proposed code for {class_name} failed syntax benchmark.")


    def identify_and_propose_improvement(self, relative_filepath: str, strategy: ImprovementStrategy):
        """
        Dispatcher method that identifies and proposes improvements based on the chosen strategy.
        """
        print(f"\nFrameworkImproveAgent: Attempting improvement for '{relative_filepath if relative_filepath else 'library-wide'}' using strategy: {strategy.name}")

        file_content = None
        if relative_filepath and relative_filepath != "--No specific file content--":
            file_content = self.read_file(relative_filepath)
            if file_content is None and strategy not in [ImprovementStrategy.NEW_PARADIGM_CUSTOM_DATA_TYPE, ImprovementStrategy.NEW_PARADIGM_LAYER_DESIGN]:
                print(f"FrameworkImproveAgent: Could not read file {relative_filepath} for strategy {strategy.name}. Aborting.")
                return
        elif strategy not in [ImprovementStrategy.NEW_PARADIGM_CUSTOM_DATA_TYPE, ImprovementStrategy.NEW_PARADIGM_LAYER_DESIGN]:
             print(f"FrameworkImproveAgent: Strategy {strategy.name} requires a valid file path. Aborting.")
             return


        if strategy == ImprovementStrategy.GENERAL_ANALYSIS:
            if not file_content: return print("General analysis needs file content.")
            self._general_analysis_improvement(relative_filepath)
        elif strategy == ImprovementStrategy.LINE_BY_LINE:
            if not file_content: return print("Line-by-line needs file content.")
            self._improve_line_by_line(relative_filepath, file_content)
        elif strategy == ImprovementStrategy.SECTION:
            if not file_content: return print("Section improvement needs file content.")
            self._improve_section_by_section(relative_filepath, file_content)
        elif strategy == ImprovementStrategy.NEW_PARADIGM_CUSTOM_DATA_TYPE:
            # This strategy might operate on a specific file or be library-wide
            self._propose_new_custom_data_type(relative_filepath, file_content if file_content else "--No specific file content--")
        elif strategy == ImprovementStrategy.NEW_PARADIGM_LAYER_DESIGN:
            print(f"Strategy {strategy.name} selected for {relative_filepath if relative_filepath else 'library'}, but not yet fully implemented.")
            # self._propose_new_layer_design(relative_filepath, file_content) # Call placeholder
        else:
            print(f"FrameworkImproveAgent: Unknown or unsupported strategy: {strategy.name}")


    def simulate_benchmark(self, original_code: str, improved_code: str, context: str = "") -> bool:
        """
        Simulates a benchmark run.
        For now, it just returns True (optimistically) if improved code is valid Python.
        Added context for better logging.
        """
        # print(f"Simulating benchmark ({context}): Original vs Improved. Assuming improvement is valid if syntax holds.") # Less verbose
        if not improved_code.strip(): # If LLM returns empty string
            print(f"Benchmark failed ({context}): Improved code is empty.")
            return False
        try:
            compile(improved_code, '<string>', 'exec')
            # print(f"Improved code syntax is valid ({context}).") # Less verbose
            return True # Optimistic
        except SyntaxError as e:
            print(f"Syntax error in LLM-generated improved code ({context}): {e}")
            return False

    def _parse_task_string(self, task_description: str) -> tuple[str, ImprovementStrategy, str | None]:
        parts = task_description.split(" using ")
        main_task_part = parts[0]
        strategy_override = None

        if len(parts) > 1:
            strategy_str = parts[1].upper()
            try:
                strategy_override = ImprovementStrategy[strategy_str]
            except KeyError:
                print(f"Warning: Unknown strategy '{parts[1]}' in task string. Will use default or main task derived strategy.")

        if main_task_part.startswith("Improve module:"):
            module_name = main_task_part.replace("Improve module:", "").strip()
            if not module_name.endswith(".py"): module_name += ".py"
            return "module", strategy_override or ImprovementStrategy.GENERAL_ANALYSIS, module_name
        elif main_task_part == "Improve all modules":
            return "all_modules", strategy_override or ImprovementStrategy.GENERAL_ANALYSIS, None
        elif main_task_part.startswith("Propose new paradigm:"):
            sub_parts = main_task_part.replace("Propose new paradigm:", "").strip().split(" for ")
            paradigm_type_str = sub_parts[0].strip().upper()
            target_name = sub_parts[1].strip() if len(sub_parts) > 1 else None # Can be a file or None (library-wide)

            # Try to map paradigm_type_str to a strategy
            # e.g. CUSTOM_DATA_TYPE -> NEW_PARADIGM_CUSTOM_DATA_TYPE
            derived_strategy = None
            try:
                derived_strategy = ImprovementStrategy[f"NEW_PARADIGM_{paradigm_type_str}"]
            except KeyError:
                 print(f"Warning: Unknown paradigm type '{paradigm_type_str}' for strategy mapping.")
                 return "unknown", strategy_override or ImprovementStrategy.GENERAL_ANALYSIS, target_name

            return "new_paradigm", strategy_override or derived_strategy, target_name

        print(f"Warning: Could not fully parse task: {task_description}. Defaulting to general module improvement if possible.")
        # Fallback for simple "Improve example_module.py"
        if main_task_part.endswith(".py"):
            return "module", strategy_override or ImprovementStrategy.GENERAL_ANALYSIS, main_task_part

        return "unknown", strategy_override or ImprovementStrategy.GENERAL_ANALYSIS, None


    def perform_task(self, task_description: str) -> None:
        """
        Performs a task, which for this agent is typically to improve the codebase
        based on a specified module and strategy.
        """
        print(f"\nFrameworkImproveAgent starting task: {task_description}")

        task_type, strategy, target_name = self._parse_task_string(task_description)

        if task_type == "module" and target_name:
            files_in_library_root = self.list_library_files()
            if target_name in files_in_library_root:
                self.identify_and_propose_improvement(relative_filepath=target_name, strategy=strategy)
            else:
                found_module = False
                # Simple check for existing subdirectories (custom_types, etc.)
                # This could be expanded if library structure becomes deeper
                for subdir_candidate in ["custom_types", "utils"]: # Example subdirs
                    potential_path = os.path.join(subdir_candidate, target_name)
                    # Need to verify if self.read_file can handle this path or if list_library_files needs to be smarter
                    # For now, assume target_name might be 'custom_types/some_type.py' already
                    # This part needs robust handling of paths if we have nested structures.
                    # Let's assume for now target_name can be "subdir/file.py"
                    if self.read_file(target_name): # If target_name itself is a relative path like "custom_types/file.py"
                         self.identify_and_propose_improvement(relative_filepath=target_name, strategy=strategy)
                         found_module = True
                         break
                if not found_module:
                     # Fallback to check common subdirectories if target_name is just a filename
                    for item in files_in_library_root: # Check top-level items that are dirs
                        item_path = os.path.join(self.shared_library_path, item)
                        if os.path.isdir(item_path):
                            if target_name in self.list_library_files(sub_directory=item):
                                self.identify_and_propose_improvement(
                                    relative_filepath=os.path.join(item, target_name),
                                    strategy=strategy
                                )
                                found_module = True
                                break
                if not found_module:
                    print(f"FrameworkImproveAgent: Module '{target_name}' not found in shared library or known subdirectories.")


        elif task_type == "all_modules":
            print(f"FrameworkImproveAgent: Attempting to improve all Python modules using strategy: {strategy.name}.")
            # This needs to be recursive or smarter to find all .py files
            # For now, simple flat + one level down from root
            processed_files = set()

            # Root .py files
            for py_file in [f for f in self.list_library_files() if f.endswith(".py")]:
                self.identify_and_propose_improvement(relative_filepath=py_file, strategy=strategy)
                processed_files.add(py_file)

            # Files in known/common subdirectories (extend as needed)
            for subdir in [d for d in self.list_library_files() if os.path.isdir(os.path.join(self.shared_library_path, d))]:
                for py_file in [f for f in self.list_library_files(sub_directory=subdir) if f.endswith(".py")]:
                    full_rel_path = os.path.join(subdir, py_file)
                    if full_rel_path not in processed_files:
                        self.identify_and_propose_improvement(relative_filepath=full_rel_path, strategy=strategy)
                        processed_files.add(full_rel_path)

        elif task_type == "new_paradigm":
            # target_name could be a file for context, or None for library-wide
            self.identify_and_propose_improvement(relative_filepath=target_name, strategy=strategy)

        else:
            print(f"FrameworkImproveAgent: Task type '{task_type}' (from '{task_description}') not understood or not implemented yet.")


if __name__ == '__main__':
    # Setup a temporary shared library for testing
    PROJECT_ROOT_FIA = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    TEST_SHARED_LIB_PATH_FIA = os.path.join(PROJECT_ROOT_FIA, 'temp_shared_library_fia_test')

    DUMMY_MODULE_FILENAME = "fia_test_module.py"
    DUMMY_MODULE_CONTENT = """
def original_function(x, y):
    # This is an original function
    z = x + y # A comment on this line
    if z > 10:
        print(f"Result {z} is greater than 10")
    else:
        print(f"Result {z} is not greater than 10") # Another comment
    return z

class MyClass:
    def __init__(self, value):
        self.value = value # Initialization

    def process(self, factor):
        # Process the value
        self.value = self.value * factor
        return self.value
"""

    # Ensure the test directory is clean before starting
    if os.path.exists(TEST_SHARED_LIB_PATH_FIA):
        import shutil
        shutil.rmtree(TEST_SHARED_LIB_PATH_FIA)
    os.makedirs(TEST_SHARED_LIB_PATH_FIA, exist_ok=True)

    with open(os.path.join(TEST_SHARED_LIB_PATH_FIA, DUMMY_MODULE_FILENAME), 'w') as f:
        f.write(DUMMY_MODULE_CONTENT)

    # Create a dummy subdir for testing "all_modules"
    DUMMY_SUBDIR = "utils_fia_test_subdir"
    DUMMY_SUBDIR_MODULE_FILENAME = "helper_fia_test.py"
    DUMMY_SUBDIR_MODULE_CONTENT = "def helper_func_fia(): return 'Helper FIA Test'"
    os.makedirs(os.path.join(TEST_SHARED_LIB_PATH_FIA, DUMMY_SUBDIR), exist_ok=True)
    with open(os.path.join(TEST_SHARED_LIB_PATH_FIA, DUMMY_SUBDIR, DUMMY_SUBDIR_MODULE_FILENAME), 'w') as f:
        f.write(DUMMY_SUBDIR_MODULE_CONTENT)

    print(f"FrameworkImproveAgent Test: Using shared library at {TEST_SHARED_LIB_PATH_FIA}")

    # Use a specific, known model for predictability if possible, or default
    llm_model_for_test = AVAILABLE_OLLAMA_MODELS[0] if AVAILABLE_OLLAMA_MODELS else None
    agent = FrameworkImproveAgent(llm_model_name=llm_model_for_test,
                                  shared_library_path=TEST_SHARED_LIB_PATH_FIA)

    test_module_path = DUMMY_MODULE_FILENAME

    print(f"\n--- Test Task: Improve module: {test_module_path} using GENERAL_ANALYSIS ---")
    agent.perform_task(f"Improve module: {test_module_path} using GENERAL_ANALYSIS")

    print(f"\n--- Test Task: Improve module: {test_module_path} using LINE_BY_LINE ---")
    agent.perform_task(f"Improve module: {test_module_path} using LINE_BY_LINE")

    print(f"\n--- Test Task: Improve module: {test_module_path} using SECTION ---")
    agent.perform_task(f"Improve module: {test_module_path} using SECTION")

    print(f"\n--- Test Task: Propose new paradigm: CUSTOM_DATA_TYPE for {test_module_path} ---")
    agent.perform_task(f"Propose new paradigm: CUSTOM_DATA_TYPE for {test_module_path}")

    print(f"\n--- Test Task: Propose new paradigm: CUSTOM_DATA_TYPE (library-wide) ---")
    agent.perform_task(f"Propose new paradigm: CUSTOM_DATA_TYPE") # No specific file, implies library-wide context

    # Test "Improve all modules" with a specific strategy
    # print(f"\n--- Test Task: Improve all modules using LINE_BY_LINE ---")
    # agent.perform_task("Improve all modules using LINE_BY_LINE")

    # Clean up
    if os.path.exists(TEST_SHARED_LIB_PATH_FIA):
        import shutil # Re-import in case it's somehow lost scope (though unlikely)
        shutil.rmtree(TEST_SHARED_LIB_PATH_FIA)
        print(f"\nCleaned up test directory: {TEST_SHARED_LIB_PATH_FIA}")

    print("\nFrameworkImproveAgent __main__ tests completed.")
