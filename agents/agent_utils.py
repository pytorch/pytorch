import os
import subprocess
import ast
import random

# Example list of available Ollama models that are good for coding tasks
AVAILABLE_OLLAMA_MODELS = [
    "codellama:latest",    # Meta's Code Llama
    "mistral-instruct:latest", # Mistral's instruct model, good for coding
    "deepseek-coder:latest", # DeepSeek Coder
]

class OllamaLLM:
    """
    A wrapper class to simulate interaction with a local Ollama-served LLM.
    In a real implementation, this class would handle HTTP requests to the Ollama API.
    """
    def __init__(self, model_name: str):
        if model_name not in AVAILABLE_OLLAMA_MODELS:
            # Fallback to a default if the specified model isn't in our known list
            print(f"Warning: Model '{model_name}' not in predefined list. Using default 'codellama:latest'.")
            self.model_name = "codellama:latest"
        else:
            self.model_name = model_name
        print(f"OllamaLLM initialized with model: {self.model_name}")

    def _send_request_to_ollama(self, prompt: str, system_message: str = None) -> str:
        """
        Placeholder for sending a request to the Ollama API.
        Enhanced to simulate layer suggestions and model code generation more effectively.
        """
        # print(f"Simulating request to Ollama model {self.model_name} with prompt: '{prompt[:100]}...'")

        # --- Priority 1: Check for Layer Suggestion Prompt ---
        suggestion_keyword = "Suggest the name of ONE suitable Python module file"
        library_keyword = "Available modules/blocks in our shared library are: "

        if suggestion_keyword in prompt and library_keyword in prompt:
            try:
                list_str_start = prompt.find(library_keyword) + len(library_keyword)
                list_str_end = prompt.find("]", list_str_start) + 1
                list_str = prompt[list_str_start:list_str_end]
                # import ast # Already at top
                available_modules = ast.literal_eval(list_str)
                if isinstance(available_modules, list) and available_modules:
                    # import random # Already at top
                    chosen_module = random.choice(available_modules)
                    # print(f"OllamaLLM (Simulated Layer Suggestion): Chose '{chosen_module}' from {available_modules}") # Verbose
                    return chosen_module
                # else: # Less verbose
                    # print(f"OllamaLLM (Simulated Layer Suggestion): Parsed empty or invalid module list: {available_modules}")
            except Exception as e:
                # print(f"OllamaLLM (Simulated Layer Suggestion): Error parsing module list from prompt: {e}.") # Verbose
                pass # Fall through to other checks if parsing fails

        # --- Priority 2: Check for Model Class Generation Prompt ---
        model_gen_keyword = "Generate a Python class for a neural network model named '"
        if model_gen_keyword in prompt:
            try:
                name_start = prompt.find(model_gen_keyword) + len(model_gen_keyword)
                name_end = prompt.find("'", name_start)
                model_name = prompt[name_start:name_end]
                if not model_name.isidentifier():
                    model_name = "DynamicGeneratedModel"

                layers_init_code = ""
                layers_str_start = prompt.find("The layer instantiations in the __init__ method should be similar to:\n")
                if layers_str_start != -1:
                    layers_str_actual_start = layers_str_start + len("The layer instantiations in the __init__ method should be similar to:\n")
                    # Try to find a sensible end for the layers block
                    layers_str_end = prompt.find("\n\nDefine the __init__ method", layers_str_actual_start)
                    if layers_str_end == -1: # if specific end not found, look for next major section or end of prompt
                        layers_str_end = prompt.find("\n\nAssume the input to forward is 'x'", layers_str_actual_start)
                    if layers_str_end == -1:
                        layers_str_end = len(prompt)

                    raw_layers_block = prompt[layers_str_actual_start:layers_str_end].strip()
                    layers_init_code = "\n".join([f"        {line.strip()}" for line in raw_layers_block.splitlines() if line.strip()])


                imports_code = ""
                imports_str_start = prompt.find("The necessary import statements are:\n")
                if imports_str_start != -1:
                    imports_str_actual_start = imports_str_start + len("The necessary import statements are:\n")
                    # Try to find a sensible end for imports block
                    imports_str_end = prompt.find("\n\nThe layer instantiations", imports_str_actual_start)
                    if imports_str_end == -1: # if specific end not found, look for next major section or end of prompt
                         imports_str_end = prompt.find("\n\nGenerate a Python class", imports_str_actual_start) # Before class def
                    if imports_str_end == -1:
                        imports_str_end = len(prompt)

                    raw_imports_block = prompt[imports_str_actual_start:imports_str_end].strip()
                    imports_code = "\n".join([line.strip() for line in raw_imports_block.splitlines() if line.strip()])
                    if imports_code: # Add newlines only if there's content
                        imports_code += "\n\n"

                num_layers = layers_init_code.count("self.layer_")
                forward_pass_lines = []
                if num_layers > 0:
                    current_var = "x"
                    for i in range(1, num_layers + 1):
                        next_var = f"out_layer{i}" if i < num_layers else "x"
                        forward_pass_lines.append(f"        {next_var} = self.layer_{i}({current_var})")
                        current_var = next_var
                    if num_layers == 1: # if only one layer, its output is the return
                         forward_pass_lines[-1] = f"        x = self.layer_1(x)" # Ensure last line assigns to x if it's the return
                    forward_pass_lines.append(f"        return x")

                else: # Default forward pass if no layers parsed
                    forward_pass_lines.append("        # No layers defined in __init__ to call in forward pass")
                    forward_pass_lines.append("        return x")

                forward_pass_code = "\n".join(forward_pass_lines)

                simulated_response = (
                    f"{imports_code.strip()}\n"
                    f"class {model_name}:\n"
                    f"    def __init__(self):\n"
                    f"{layers_init_code if layers_init_code else '        pass'}\n"
                    f"    def forward(self, x):\n"
                    f"{forward_pass_code}\n"
                )
                # print(f"OllamaLLM (Simulated Model Code Gen): Generated class {model_name}") # Verbose
                return simulated_response.strip()
            except Exception as e:
                # print(f"OllamaLLM (Simulated Model Code Gen): Error parsing model name or layers: {e}") # Verbose
                pass # Fall through to generic simulation

        # --- Priority 3: Fallback / More Generic Simulation ---
        if "analyze code" in prompt.lower():
            return "This code appears to be functional but could benefit from more detailed comments and error handling. No specific issues found."
        else: # General code generation or unknown prompts
            # print(f"OllamaLLM (Fallback Simulation) for prompt: '{prompt[:50]}...'") # Verbose
            return "class FallbackGeneratedClass:\n    def __init__(self):\n        pass\n    def forward(self, x):\n        # Default fallback response\n        return x"

    def generate_code(self, prompt: str, context: str = "") -> str:
        """
        Simulates generating code based on a prompt and optional context.
        """
        full_prompt = f"{context}\n\nGenerate code based on the following prompt: {prompt}"
        # In a real scenario, you might have a specific system message for code generation
        system_message = "You are an expert coding assistant. Generate only Python code as a direct response to the prompt, without any surrounding text or explanations."
        # print(f"OllamaLLM ({self.model_name}): Generating code for prompt - '{prompt}'") # Less verbose
        return self._send_request_to_ollama(full_prompt, system_message)

    def analyze_code(self, code_content: str, analysis_prompt: str) -> str:
        """
        Simulates analyzing a piece of code based on a specific analysis prompt.
        """
        full_prompt = f"Analyze the following code snippet:\n```python\n{code_content}\n```\n\nAnalysis prompt: {analysis_prompt}"
        # System message could guide the analysis style
        system_message = "You are an expert code analyst. Provide a concise analysis based on the prompt."
        # print(f"OllamaLLM ({self.model_name}): Analyzing code with prompt - '{analysis_prompt}'") # Less verbose
        return self._send_request_to_ollama(full_prompt, system_message)

if __name__ == '__main__':
    print("Available Ollama Models:", AVAILABLE_OLLAMA_MODELS)

    if not AVAILABLE_OLLAMA_MODELS:
        print("No Ollama models defined. Skipping tests.")
    else:
        llm_instance = OllamaLLM(model_name=AVAILABLE_OLLAMA_MODELS[0])

        print("\n--- Test: Generic Code Generation ---")
        generic_code_prompt = "Create a Python function that multiplies two numbers."
        generated_code = llm_instance.generate_code(prompt=generic_code_prompt)
        print(f"Prompt: {generic_code_prompt}\nGenerated Code:\n{generated_code}")

        print("\n--- Test: Code Analysis ---")
        analysis_prompt_test = "Identify potential improvements in this code."
        code_to_analyze = "def my_func(a,b): return a+b"
        analysis_result = llm_instance.analyze_code(code_content=code_to_analyze, analysis_prompt=analysis_prompt_test)
        print(f"Code: {code_to_analyze}\nPrompt: {analysis_prompt_test}\nAnalysis Result:\n{analysis_result}")

        print("\n--- Test: Simulated Layer Suggestion (Successful Parse) ---")
        layer_suggestion_prompt_success = (
            "You are a machine learning model architect. "
            "The input data is described as: 'tabular data'.\n"
            "The current model has these layers (in order): [].\n"
            "Available modules/blocks in our shared library are: ['linear_layer.py', 'relu_activation.py', 'output_layer.py'].\n"
            "Suggest the name of ONE suitable Python module file from the shared library. Only provide the module filename."
        )
        suggested_layer_success = llm_instance.generate_code(prompt=layer_suggestion_prompt_success) # generate_code calls _send_request_to_ollama
        print(f"Prompt contains library list. Suggested Layer: {suggested_layer_success}")
        assert suggested_layer_success in ['linear_layer.py', 'relu_activation.py', 'output_layer.py']

        print("\n--- Test: Simulated Layer Suggestion (Malformed List in Prompt) ---")
        layer_suggestion_prompt_malformed = (
            "You are a machine learning model architect. "
            "Available modules/blocks in our shared library are: ['module_a.py', not_a_string, 'module_b.py'].\n" # Malformed list
            "Suggest the name of ONE suitable Python module file."
        )
        suggested_layer_malformed = llm_instance.generate_code(prompt=layer_suggestion_prompt_malformed)
        print(f"Prompt contains malformed list. Suggested Layer/Response: {suggested_layer_malformed}")
        # In this case, it should fall back to the default simulation, likely returning a class definition

        print("\n--- Test: Simulated Layer Suggestion (Keywords Missing) ---")
        layer_suggestion_prompt_missing_keywords = (
            "Available modules: ['mod1.py', 'mod2.py']. Suggest one." # Missing the full keywords
        )
        suggested_layer_missing = llm_instance.generate_code(prompt=layer_suggestion_prompt_missing_keywords)
        print(f"Prompt missing keywords. Suggested Layer/Response: {suggested_layer_missing}")
        # Should also fall back to default simulation

        print("\n--- Test: Model Not in List (Warning) ---")
        llm_other_instance = OllamaLLM(model_name="non_existent_model:latest") # Should print a warning
        generated_code_other = llm_other_instance.generate_code(prompt="Create a class for a user.")
        print(f"Prompt for non-existent model. Generated Code:\n{generated_code_other}")

        print("\n--- Test: Simulated Model Code Generation (Dynamic Name & Clean Code) ---")
        model_gen_prompt = (
            "Generate a Python class for a neural network model named 'MyDynamicTestModel'.\n"
            "The necessary import statements are:\nfrom shared_library.layer_module import LayerModule\nimport torch\n\n"
            "The layer instantiations in the __init__ method should be similar to:\n"
            "            self.layer_1 = LayerModule()\n"
            "            self.layer_2 = LayerModule()\n\n"
            "Define the __init__ method to initialize these layers and a forward() method. "
        )
        generated_model_code = llm_instance.generate_code(prompt=model_gen_prompt)
        print(f"Prompt for model gen. Generated Code:\n{generated_model_code}")
        assert "class MyDynamicTestModel:" in generated_model_code, "Dynamic class name not found"
        assert "from shared_library.layer_module import LayerModule" in generated_model_code, "Import statement missing"
        assert "import torch" in generated_model_code, "Second import statement missing"
        assert "self.layer_1 = LayerModule()" in generated_model_code, "Layer 1 init missing"
        assert "self.layer_2 = LayerModule()" in generated_model_code, "Layer 2 init missing"
        assert "forward(self, x):" in generated_model_code, "Forward method missing"
        assert "out_layer1 = self.layer_1(x)" in generated_model_code, "Forward pass layer 1 call incorrect"
        assert "x = self.layer_2(out_layer1)" in generated_model_code, "Forward pass layer 2 call incorrect"
        assert "//" not in generated_model_code, "Problematic comment '//' found"

        print("\n--- Test: Simulated Model Code Generation (Simpler, one layer) ---")
        model_gen_prompt_simple = (
            "Generate a Python class for a neural network model named 'MySimpleModel'.\n"
            "The necessary import statements are:\nfrom shared_library.simple_layer import SimpleLayer\n\n"
            "The layer instantiations in the __init__ method should be similar to:\n"
            "            self.layer_1 = SimpleLayer()\n\n"
            "Define the __init__ method to initialize these layers and a forward() method. "
        )
        generated_model_code_simple = llm_instance.generate_code(prompt=model_gen_prompt_simple)
        print(f"Prompt for simple model gen. Generated Code:\n{generated_model_code_simple}")
        assert "class MySimpleModel:" in generated_model_code_simple, "Simple model class name error"
        assert "from shared_library.simple_layer import SimpleLayer" in generated_model_code_simple, "Simple model import error"
        assert "self.layer_1 = SimpleLayer()" in generated_model_code_simple, "Simple model layer init error"
        assert "x = self.layer_1(x)" in generated_model_code_simple, "Simple model forward pass error for single layer"
        assert "return x" in generated_model_code_simple, "Simple model return error"

        print("\n--- Test: Simulated Model Code Generation (No specific layers in prompt) ---")
        model_gen_prompt_no_layers = (
            "Generate a Python class for a neural network model named 'NoLayerModel'."
        )
        generated_model_code_no_layers = llm_instance.generate_code(prompt=model_gen_prompt_no_layers)
        print(f"Prompt for no-layer model. Generated Code:\n{generated_model_code_no_layers}")
        assert "class NoLayerModel:" in generated_model_code_no_layers
        assert "def __init__(self):\n        pass" in generated_model_code_no_layers # Expecting pass in init
        assert "def forward(self, x):\n        # No layers defined in __init__ to call in forward pass\n        return x" in generated_model_code_no_layers

    print("\nagent_utils.py tests completed.")
