#%%     **************** Training and validation *****************
# This script is designed to run the SAM2.1 training and validation process.
import os
import yaml
import subprocess
import sys

# Set working directory to the same one you use in terminal
os.chdir()

# Define configuration relative to that directory
yaml_file = "configs/sam2.1_training/sam_train_val_json_win.yaml"

def load_yaml_config(yaml_path):
    try:
        # Construct the absolute path
        abs_path = os.path.join("insert_path", yaml_path)
        with open(abs_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cfg
    except Exception as e:
        print(f"Error loading YAML: {e}")
        return None

def run_training(config_path):
    print(f"Launching training with config: {config_path}")
    command = [
        "python", "training/train.py",
        "-c", config_path,
        "--use-cluster", "0",
        "--num-gpus", "1"
    ]

    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1 # Line buffered
    ) as process:
        # Print stdout as it comes
        for line in process.stdout:
            print(line, end='')

        # Wait for the process to finish and get the exit code
        process.wait()
        if process.returncode != 0:
            # If it failed, print the error
            error_output = process.stderr.read()
            print("--- ERROR DURING TRAINING ---", file=sys.stderr)
            print(error_output, file=sys.stderr)
            raise subprocess.CalledProcessError(process.returncode, command,
                                                output=error_output)


if __name__ == "__main__":
    config_path = yaml_file  
    config = load_yaml_config(config_path)
    print("Loaded Configuration:", config)
    run_training(config_path)


