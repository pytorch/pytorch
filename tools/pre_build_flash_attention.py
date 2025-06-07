

import json
import subprocess
import os

def get_compile_commands() -> list[str]:
    with open("build/compile_commands.json", "r") as file:
        return json.load(file)

def main() -> None:
    compile_commands = get_compile_commands()
    for command in compile_commands:
        output = command["output"]

        if "flash-attention" in output:
            print(f"Compiling {output} with command: {command['command']}")
            os.makedirs(os.path.dirname(output), exist_ok=True)
            res = subprocess.run(
                command["command"].split(),
            )
            if res.returncode != 0:
                print(f"Failed to compile {output}, command: {command['command']}")


if __name__ == "__main__":
    main()
