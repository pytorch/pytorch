import os
from ..base_agent import BaseAgent, DEFAULT_SHARED_LIBRARY_PATH
from ..agent_utils import AVAILABLE_OLLAMA_MODELS

class ModelBuilderAgent(BaseAgent):
    """
    An agent responsible for constructing machine learning models using modules
    from the shared library based on input data characteristics.
    """
    def __init__(self,
                 llm_model_name: str = None,
                 shared_library_path: str = None,
                 initial_models_count: int = 3,
                 top_n_initial_models_to_select: int = 1,
                 max_model_layers: int = 4,
                 max_iterations_no_improvement: int = 2,
                 performance_drop_threshold: float = 0.05, # e.g., if acc drops by >5% from previous step
                 min_performance_improvement: float = 0.01 # Min improvement to reset no_improvement_counter
                ):
        super().__init__(llm_model_name=llm_model_name, shared_library_path=shared_library_path)

        self.initial_models_count = initial_models_count
        self.top_n_initial_models_to_select = top_n_initial_models_to_select
        self.max_model_layers = max_model_layers
        self.max_iterations_no_improvement = max_iterations_no_improvement
        self.performance_drop_threshold = performance_drop_threshold
        self.min_performance_improvement = min_performance_improvement
        self.primary_metric = "accuracy" # Metric to optimize for

        # print(f"ModelBuilderAgent initialized with LLM: {self.llm.model_name}, Library: {self.shared_library_path}")
        # print(f"Configuration: initial_models={self.initial_models_count}, top_n={self.top_n_initial_models_to_select}, max_layers={self.max_model_layers}")
        # print(f"max_no_improvement_iters={self.max_iterations_no_improvement}, perf_drop_thresh={self.performance_drop_threshold}, min_perf_improve={self.min_performance_improvement}")

    def suggest_model_layer_or_block(self, input_data_description: str, current_model_layers: list[str] = None) -> str | None:
        """
        Suggests a layer or block from the shared library suitable for the given data and current model.
        """
        print(f"\nModelBuilderAgent: Suggesting layer/block for data: '{input_data_description}'")
        if current_model_layers is None:
            current_model_layers = []

        library_modules = self.list_library_files()
        # Filter for .py files and remove extension for cleaner suggestions, or list as is
        # For now, just list them. A more advanced agent might categorize them.

        prompt = (
            f"You are a machine learning model architect. "
            f"The input data is described as: '{input_data_description}'.\n"
            f"The current model has these layers (in order): {current_model_layers}.\n"
            f"Available modules/blocks in our shared library are: {library_modules}.\n"
            f"Suggest the name of ONE suitable Python module file (e.g., 'linear_layer.py', 'conv_block.py') "
            f"from the shared library to add as the next component of the model. "
            f"If no suitable layer is found or the model seems complete, suggest 'None'. "
            f"Only provide the module filename or 'None'."
        )

        suggested_module = self.llm.generate_code(prompt=prompt).strip()

        # Basic validation of suggestion
        if suggested_module == "None" or suggested_module.lower() == "none":
            print("ModelBuilderAgent: LLM suggested no further layers.")
            return None

        # Further clean up common LLM artifacts if any (like "module: filename.py")
        if ":" in suggested_module:
            suggested_module = suggested_module.split(":")[-1].strip()
        if suggested_module.endswith(".py") and suggested_module in library_modules:
            print(f"ModelBuilderAgent: LLM suggested module: '{suggested_module}'")
            return suggested_module
        elif not suggested_module.endswith(".py") and f"{suggested_module}.py" in library_modules:
            suggested_module_py = f"{suggested_module}.py"
            print(f"ModelBuilderAgent: LLM suggested module (added .py): '{suggested_module_py}'")
            return suggested_module_py
        else:
            print(f"ModelBuilderAgent: LLM suggested '{suggested_module}', which is not a valid or available module. Library: {library_modules}")
            # Fallback or further processing could happen here
            return None


    def construct_model_code(self, model_layer_sequence: list[str], model_name: str = "MyCustomModel") -> str | None:
        """
        Generates Python code for a model composed of the given sequence of layers/blocks.
        Each item in model_layer_sequence is expected to be a module filename from the shared library.
        """
        print(f"\nModelBuilderAgent: Constructing model code for sequence: {model_layer_sequence}")
        if not model_layer_sequence:
            print("ModelBuilderAgent: No layers provided to construct the model.")
            return None

        # For each layer, we might want to fetch its (simulated) interface or a summary
        # For now, we'll just use the names in the generation prompt.

        layer_instantiations = []
        import_statements = set() # To avoid duplicate imports

        for i, layer_module_name in enumerate(model_layer_sequence):
            module_name_no_ext = layer_module_name.replace(".py", "")
            # Assume class name is CamelCase version of module name, e.g., linear_layer -> LinearLayer
            class_name = "".join(word.capitalize() for word in module_name_no_ext.split('_'))

            import_statements.add(f"from shared_library.{module_name_no_ext} import {class_name}")
            layer_instantiations.append(f"            self.layer_{i+1} = {class_name}() # Placeholder for actual args")

        imports_str = "\n".join(sorted(list(import_statements)))
        layers_str = "\n".join(layer_instantiations)

        prompt = (
            f"Generate a Python class for a neural network model named '{model_name}'.\n"
            f"The model should use the following layers/blocks from our shared library, in this sequence:\n"
            f"{model_layer_sequence}\n\n"
            f"The necessary import statements are:\n{imports_str}\n\n"
            f"The layer instantiations in the __init__ method should be similar to:\n{layers_str}\n\n"
            f"Define the __init__ method to initialize these layers and a forward() method to pass data through them sequentially. "
            f"Assume the input to forward is 'x'. The output of one layer is the input to the next. "
            f"Provide only the complete Python code for the class. Ensure it's a valid PyTorch-like model structure."
        )

        model_code = self.llm.generate_code(prompt=prompt)
        print(f"ModelBuilderAgent: LLM generated model code:\n{model_code}")

        # Basic validation: check if the generated code contains the class name
        if f"class {model_name}" in model_code:
            return model_code
        else:
            print("ModelBuilderAgent: LLM failed to generate a valid class structure for the model.")
            # Return the raw code anyway for debugging if needed, or handle more gracefully
            return model_code # Or None if strict failure is preferred

    def simulate_model_training_and_evaluation(self, model_code: str, data_description: str) -> dict:
        """
        Simulates the training and evaluation of the generated model code.
        In a real system, this would involve:
        1. Loading/preparing data.
        2. Instantiating the model from model_code.
        3. A training loop.
        4. An evaluation loop using predefined or custom metrics.
        """
        print(f"\nModelBuilderAgent: Simulating training & evaluation for model with data: '{data_description}'")
        # print(f"Model Code:\n{model_code[:200]}...") # Print snippet

        # For now, just simulate a successful evaluation with some dummy metrics
        # A more advanced simulation could check syntax of model_code
        try:
            # Check if the model code is syntactically valid Python
            compile(model_code, '<string>', 'exec')
            print("Model code syntax is valid.")
            # Simulate some metrics
            metrics = {
                "accuracy": 0.75 + hash(model_code) % 20 / 100.0, # Make it slightly variable
                "loss": 0.5 - hash(model_code) % 10 / 100.0,
                "custom_metric_1": 0.80 + hash(model_code) % 15 / 100.0,
            }
            print(f"Simulated metrics: {metrics}")
            return {"success": True, "metrics": metrics, "message": "Simulated training and evaluation complete."}
        except SyntaxError as e:
            print(f"Syntax error in generated model code during simulation: {e}")
            return {"success": False, "metrics": {}, "message": f"SyntaxError: {e}"}

    def perform_task(self, task_description: str) -> None:
        """
        Performs the task of building a model by first generating and evaluating
        a set of initial single-layer models, then iteratively building upon the best ones.
        Example task_description: "Build model for tabular data, features: 10, target: binary"
        """
        print(f"\nModelBuilderAgent starting task: {task_description}")

        if not task_description.startswith("Build model for"):
            print(f"ModelBuilderAgent: Task '{task_description}' not understood.")
            return

        data_desc_for_llm = task_description.replace("Build model for ", "")

        # --- Stage 1: Generate and Evaluate Initial Single-Layer Models ---
        initial_models_performance = [] # Stores tuples of (primary_metric_value, model_code, layer_sequence, eval_results)

        print(f"\n--- Stage 1: Generating {self.initial_models_count} Initial Single-Layer Models ---")
        available_library_modules = self.list_library_files()
        if not available_library_modules:
            print("ModelBuilderAgent: Shared library is empty. Cannot suggest layers.")
            return

        used_initial_layers = set()

        for i in range(self.initial_models_count):
            print(f"\nGenerating Initial Model {i+1}/{self.initial_models_count}...")

            suggested_layer_module = None
            for _ in range(5):
                candidate_layer = self.suggest_model_layer_or_block(
                    input_data_description=data_desc_for_llm,
                    current_model_layers=[]
                )
                if candidate_layer and candidate_layer not in used_initial_layers:
                    suggested_layer_module = candidate_layer
                    used_initial_layers.add(suggested_layer_module)
                    break
                elif candidate_layer and candidate_layer in used_initial_layers:
                    print(f"ModelBuilderAgent: Layer '{candidate_layer}' already used for an initial model. Trying again.")

            if not suggested_layer_module:
                suggested_layer_module = self.suggest_model_layer_or_block(
                    input_data_description=data_desc_for_llm, current_model_layers=[]
                )
                if not suggested_layer_module:
                    print("ModelBuilderAgent: LLM failed to suggest an initial layer. Skipping this initial model.")
                    continue
                print(f"ModelBuilderAgent: Could not find a unique initial layer, proceeding with potentially repeated: {suggested_layer_module}")

            model_name = f"InitialModel_{i+1}_{suggested_layer_module.replace('.py','')}"
            model_code = self.construct_model_code(
                model_layer_sequence=[suggested_layer_module],
                model_name=model_name
            )

            if not model_code:
                print(f"ModelBuilderAgent: Failed to construct code for initial model with layer {suggested_layer_module}. Skipping.")
                continue

            print(f"Simulating evaluation for {model_name} with layer {suggested_layer_module}...")
            eval_results = self.simulate_model_training_and_evaluation(
                model_code=model_code,
                data_description=data_desc_for_llm
            )

            if eval_results["success"] and self.primary_metric in eval_results["metrics"]:
                metric_value = eval_results["metrics"][self.primary_metric]
                initial_models_performance.append(
                    (metric_value, model_code, [suggested_layer_module], eval_results)
                )
                print(f"Initial Model {model_name} (Layer: {suggested_layer_module}): {self.primary_metric} = {metric_value:.4f}")
            else:
                print(f"ModelBuilderAgent: Evaluation failed or primary metric missing for initial model with {suggested_layer_module}.")

        # Sort initial models by primary metric in descending order (higher is better)
        initial_models_performance.sort(key=lambda x: x[0], reverse=True)

        print(f"\n--- Summary of Initial Models (Sorted by {self.primary_metric}) ---")
        for idx, (metric, _, layers, _) in enumerate(initial_models_performance):
            print(f"  Rank {idx+1}: Layers: {layers}, {self.primary_metric}: {metric:.4f}")

        # --- Stage 2: Iterative Model Building ---
        # Select the top N models to form the starting points for different model lineages
        lineages_to_evolve = []
        if not initial_models_performance:
            print("ModelBuilderAgent: No initial models were successfully generated to select from. Stopping.")
            return

        selected_for_evolution = initial_models_performance[:self.top_n_initial_models_to_select]

        print(f"\n--- Selected Top {len(selected_for_evolution)} Initial Model(s) for Evolution ---")
        for i, (metric, model_code, layers, eval_res) in enumerate(selected_for_evolution):
            print(f"  Lineage {i+1} Starting Point - Layers: {layers}, Initial {self.primary_metric}: {metric:.4f}")
            # Store all relevant info for the lineage's starting point
            lineages_to_evolve.append({
                "lineage_id": f"lineage_{i+1}",
                "current_layers": list(layers), # Ensure it's a mutable copy
                "current_code": model_code,
                "current_performance": metric,
                "current_eval_results": eval_res,
                "history": [{
                    "layers": list(layers),
                    "performance": metric,
                    "action": "initial_selection"
                }],
                "iterations_no_improvement": 0,
                "best_performance_in_lineage": metric,
                "best_layers_in_lineage": list(layers),
                "best_code_in_lineage": model_code,
                "active": True # This lineage is currently active for improvement
            })

        if not lineages_to_evolve:
            print("ModelBuilderAgent: Could not select any lineages to evolve. Stopping.")
            return

        print(f"\n--- Stage 2: Iterative Layer Addition ---")
        final_models_from_lineages = []

        for lineage_idx, lineage in enumerate(lineages_to_evolve):
            if not lineage["active"]:
                print(f"Skipping inactive {lineage['lineage_id']}.")
                final_models_from_lineages.append({
                    "lineage_id": lineage["lineage_id"],
                    "layers": lineage["best_layers_in_lineage"],
                    "code": lineage["best_code_in_lineage"],
                    "performance": lineage["best_performance_in_lineage"],
                    "reason_stopped": "Marked inactive before iteration."
                })
                continue

            print(f"\nProcessing {lineage['lineage_id']} (Rank {lineage_idx+1} from initial selection)")
            print(f"  Starting with Layers: {lineage['current_layers']}, Performance: {lineage['current_performance']:.4f}")

            for layer_number in range(len(lineage["current_layers"]) + 1, self.max_model_layers + 1):
                print(f"  {lineage['lineage_id']}: Attempting to add Layer #{layer_number} (current total layers: {len(lineage['current_layers'])})")

                if lineage["iterations_no_improvement"] >= self.max_iterations_no_improvement:
                    print(f"  {lineage['lineage_id']}: Stopping early for this lineage due to reaching max iterations ({self.max_iterations_no_improvement}) without improvement.")
                    lineage["active"] = False
                    break

                # --- Suggest, Construct, and Evaluate New Layer Variant (Step 6) ---
                print(f"    {lineage['lineage_id']}: Suggesting next layer (current layers: {len(lineage['current_layers'])}, target layer count for model: {layer_number})")

                suggested_new_layer_module = self.suggest_model_layer_or_block(
                    input_data_description=data_desc_for_llm,
                    current_model_layers=lineage["current_layers"]
                )

                if not suggested_new_layer_module:
                    print(f"    {lineage['lineage_id']}: LLM suggested no further layers or failed to suggest. Stopping this lineage.")
                    lineage["active"] = False
                    lineage["reason_stopped"] = "LLM suggested no more layers."
                    break

                print(f"    {lineage['lineage_id']}: LLM suggested new layer: {suggested_new_layer_module}")

                next_layer_sequence = list(lineage["current_layers"]) + [suggested_new_layer_module]
                new_model_name = f"{lineage['lineage_id']}_L{len(next_layer_sequence)}_{suggested_new_layer_module.replace('.py','')}"

                new_model_code = self.construct_model_code(
                    model_layer_sequence=next_layer_sequence,
                    model_name=new_model_name
                )

                if not new_model_code:
                    print(f"    {lineage['lineage_id']}: Failed to construct model code for sequence {next_layer_sequence}. Incrementing no_improvement counter.")
                    lineage["iterations_no_improvement"] += 1
                    continue

                print(f"    {lineage['lineage_id']}: Evaluating new model variant with layers: {next_layer_sequence}")
                eval_results_new_variant = self.simulate_model_training_and_evaluation(
                    model_code=new_model_code,
                    data_description=data_desc_for_llm
                )

                # --- Step 7: Accept or Reject New Layer ---
                # lineage.pop('last_attempted_variant', None) # Not needed as we directly use results

                if eval_results_new_variant["success"]:
                    new_performance = eval_results_new_variant["metrics"].get(self.primary_metric, 0.0)
                    previous_performance = lineage['current_performance']

                    print(f"      Comparing: New Perf ({new_performance:.4f}) vs Prev Perf ({previous_performance:.4f}). Drop threshold: {self.performance_drop_threshold:.4f}")

                    if new_performance >= previous_performance - self.performance_drop_threshold:
                        print(f"    {lineage['lineage_id']}: Layer {suggested_new_layer_module} ACCEPTED.")
                        lineage['current_layers'] = list(next_layer_sequence)
                        lineage['current_code'] = new_model_code
                        lineage['current_performance'] = new_performance
                        lineage['current_eval_results'] = eval_results_new_variant
                        lineage['history'].append({
                            "layers": list(lineage['current_layers']),
                            "performance": new_performance,
                            "action": f"accepted_layer_{suggested_new_layer_module}"
                        })

                        if new_performance > lineage['best_performance_in_lineage']:
                            print(f"      New best performance for {lineage['lineage_id']}: {new_performance:.4f} (old best: {lineage['best_performance_in_lineage']:.4f})")
                            lineage['best_performance_in_lineage'] = new_performance
                            lineage['best_layers_in_lineage'] = list(lineage['current_layers'])
                            lineage['best_code_in_lineage'] = lineage['current_code']

                        if new_performance >= previous_performance + self.min_performance_improvement:
                            print(f"      Performance improved significantly. Resetting no_improvement_counter for {lineage['lineage_id']}.")
                            lineage['iterations_no_improvement'] = 0
                        else:
                            lineage['iterations_no_improvement'] += 1
                            print(f"      Performance accepted (stable or minor drop/improvement). No_improvement_counter for {lineage['lineage_id']}: {lineage['iterations_no_improvement']}.")
                    else:
                        print(f"    {lineage['lineage_id']}: Layer {suggested_new_layer_module} REJECTED due to significant performance drop ({new_performance:.4f} vs prev {previous_performance:.4f}).")
                        lineage['iterations_no_improvement'] += 1
                        lineage['history'].append({
                            "layers": list(next_layer_sequence),
                            "performance": new_performance,
                            "action": f"rejected_layer_perf_drop_{suggested_new_layer_module}"
                        })

                else:
                    print(f"    {lineage['lineage_id']}: Layer {suggested_new_layer_module} REJECTED due to evaluation failure for variant {new_model_name}.")
                    lineage['iterations_no_improvement'] += 1
                    lineage['history'].append({
                        "layers": list(next_layer_sequence),
                        "performance": 0.0,
                        "action": f"rejected_layer_eval_fail_{suggested_new_layer_module}"
                    })
                # --- End of Step 7 Logic ---

            if lineage["active"]:
                print(f"  {lineage['lineage_id']}: Finished attempting layer additions. Final layers for this lineage: {lineage['current_layers']}")

            final_models_from_lineages.append({
                "lineage_id": lineage["lineage_id"],
                "layers": lineage["best_layers_in_lineage"],
                "code": lineage["best_code_in_lineage"],
                "performance": lineage["best_performance_in_lineage"],
                "reason_stopped": "Max layers reached or iterations no improvement." if lineage["active"] else lineage.get("reason_stopped", "Stopped early due to no improvement.")
            })
            lineage["active"] = False

        # --- Finalize and Report Best Model(s) ---
        print("\n--- Overall Model Building Summary ---")

        if not final_models_from_lineages:
            print("  No successful model lineages were evolved.")
            print("\nModelBuilderAgent task completed: No models produced.")
            return

        # Sort all found best models from lineages by performance
        final_models_from_lineages.sort(key=lambda x: x["performance"], reverse=True)

        print("\nTop Performing Models from All Lineages:")
        for i, model_summary in enumerate(final_models_from_lineages):
            print(f"  Rank {i+1}:")
            print(f"    Lineage ID: {model_summary['lineage_id']}")
            print(f"    Best Layers: {model_summary['layers']}")
            print(f"    Performance ({self.primary_metric}): {model_summary['performance']:.4f}")
            print(f"    Reason Lineage Stopped: {model_summary['reason_stopped']}")
            # print(f"    Best Code Snippet:\n{model_summary['code'][:200]}...") # Optional: print snippet
            if i == 0: # Highlight the overall best
                print("    --- This is the OVERALL BEST MODEL FOUND ---")

        overall_best_model = final_models_from_lineages[0]
        print(f"\nOverall Best Model Details:")
        print(f"  Lineage ID: {overall_best_model['lineage_id']}")
        print(f"  Achieved Performance ({self.primary_metric}): {overall_best_model['performance']:.4f}")
        print(f"  Layer Sequence: {overall_best_model['layers']}")
        print(f"  Generated Code for Best Model:\n```python\n{overall_best_model['code']}\n```")

        print("\nModelBuilderAgent task completed.")


if __name__ == '__main__':
    PROJECT_ROOT_MBA = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    TEST_SHARED_LIB_PATH_MBA = os.path.join(PROJECT_ROOT_MBA, 'temp_shared_library_mba_test_iterative') # New test dir name

    dummy_layers_content = {
        "linear_layer.py": "class LinearLayer:\n  def __init__(self, i=0, o=0): pass\n  def forward(self, x): return x",
        "relu_activation.py": "class ReluActivation:\n  def __init__(self):pass\n  def forward(self, x): return x",
        "conv_block.py": "class ConvBlock:\n  def __init__(self): pass\n  def forward(self, x): return x",
        "output_layer.py": "class OutputLayer:\n  def __init__(self): pass\n  def forward(self, x): return x",
        "dropout_layer.py": "class DropoutLayer:\n  def __init__(self): pass\n  def forward(self, x): return x"
    }

    if os.path.exists(TEST_SHARED_LIB_PATH_MBA):
        import shutil
        shutil.rmtree(TEST_SHARED_LIB_PATH_MBA)
    os.makedirs(TEST_SHARED_LIB_PATH_MBA, exist_ok=True)

    for fname, content in dummy_layers_content.items():
        with open(os.path.join(TEST_SHARED_LIB_PATH_MBA, fname), 'w') as f:
            f.write(content)

    print(f"ModelBuilderAgent Test (Iterative Stage 1): Using shared library at {TEST_SHARED_LIB_PATH_MBA}")

    llm_to_use = AVAILABLE_OLLAMA_MODELS[0] if AVAILABLE_OLLAMA_MODELS else None
    agent = ModelBuilderAgent(
        llm_model_name=llm_to_use,
        shared_library_path=TEST_SHARED_LIB_PATH_MBA,
        initial_models_count=3,
        top_n_initial_models_to_select=2,
        max_model_layers=4,
        max_iterations_no_improvement=2
    )

    print("\n--- Test: Build model for tabular data (Initial Generation Phase) ---")
    agent.perform_task("Build model for tabular data, features: 10, target: binary classification")

    if os.path.exists(TEST_SHARED_LIB_PATH_MBA):
        import shutil
        shutil.rmtree(TEST_SHARED_LIB_PATH_MBA)
        print(f"\nCleaned up test directory: {TEST_SHARED_LIB_PATH_MBA}")

    print("\nModelBuilderAgent initial generation tests completed.")
