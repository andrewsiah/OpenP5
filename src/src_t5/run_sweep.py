import wandb
import subprocess
import os
import yaml
from pathlib import Path

def load_sweep_config(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def train():
    run = wandb.init()
    config = wandb.config
    
    # Get path to main.py relative to this file
    main_path = Path(__file__).parent / "main.py"
    
    # Convert config to command line arguments
    cmd = ["uv", "run", str(main_path)]
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.append(f"--{key}")
            cmd.append(str(value))
    
    print(f"Running command: {' '.join(cmd)}")  # Print the full command
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    # Stream both stdout and stderr
    while True:
        stdout_line = process.stdout.readline()
        stderr_line = process.stderr.readline()
        
        if stdout_line == '' and stderr_line == '' and process.poll() is not None:
            break
            
        if stdout_line:
            line = stdout_line.strip()
            print(line)
            wandb.log({"output": line})
            
        if stderr_line:
            line = stderr_line.strip()
            print(f"ERROR: {line}")
            wandb.log({"error": line})
    
    process.stdout.close()
    process.stderr.close()
    return_code = process.wait()
    
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)

if __name__ == "__main__":
    # Get the root directory of the project
    root_dir = Path(__file__).parent.parent.parent
    
    # Path to sweep config
    sweep_config_path = root_dir / "sweeps" / "mix_cds_sequential.yaml"
    sweep_config = load_sweep_config(sweep_config_path)
    
    sweep_id = wandb.sweep(sweep_config, project="OpenP5")
    wandb.agent(sweep_id, function=train)