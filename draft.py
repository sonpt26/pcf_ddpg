import os
from network import NetworkEnv
from pathlib import Path

directory_path = "./setting"
specific_dir = Path(directory_path)


# List all directories in the specified directory
folders = [
    name
    for name in os.listdir(directory_path)
    if os.path.isdir(os.path.join(directory_path, name))
]

# Print the list of folders
print("Folders in directory:", directory_path)
for folder in folders:
    gen_setting = specific_dir / folder / "generator.yaml"
    proc_setting = specific_dir / folder / "processor.yaml"    
    env = NetworkEnv(gen_setting, proc_setting)

