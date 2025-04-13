# Please contact the PyTorch docs team before decreasing this number.
# It is important that our docs build in a reasonable amount of time.
NB_EXECUTION_TIMEOUT_SECONDS = 30

def handler(app, env):
    for page, data in env.nb_metadata.items():
        runtime = data['exec_data']['runtime']
        if runtime > NB_EXECUTION_TIMEOUT_SECONDS:
            raise RuntimeError(
                f"Page {page} took too long ({runtime}s) to execute, we require that all pages "
                f"execute in less than {NB_EXECUTION_TIMEOUT_SECONDS}s. "
                f"If your code doesn't need to be executable then consider removing "
                f"the code-cell directive. "
                f"Full runtime breakdown: {env.nb_metadata}")

def setup(app):
    app.connect("env-updated", handler)
