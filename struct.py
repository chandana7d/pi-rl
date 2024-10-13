import os

# Define the folder structure as a list of paths
folder_structure = [
    "src/pi/core/agents",
    "src/pi/core/agents/value_based",
    "src/pi/core/agents/policy_based",
    "src/pi/core/agents/actor_critic",
    "src/pi/core/agents/meta_learning"
]

# Define the files to create in each directory
files_to_create = {
    "src/pi/core/agents": ["__init__.py", "base_agent.py"],
    "src/pi/core/agents/value_based": [
        "__init__.py",
        "dqn_agent.py",
        "double_dqn_agent.py",
        "dueling_dqn_agent.py",
        "rainbow_agent.py"
    ],
    "src/pi/core/agents/policy_based": [
        "__init__.py",
        "ppo_agent.py",
        "vpg_agent.py",
        "sac_agent.py"
    ],
    "src/pi/core/agents/actor_critic": [
        "__init__.py",
        "a2c_agent.py",
        "a3c_agent.py",
        "sac_agent.py"
    ],
    "src/pi/core/agents/meta_learning": [
        "__init__.py",
        "maml_agent.py",
        "rl2_agent.py"
    ]
}

# Create the folder structure
for folder in folder_structure:
    os.makedirs(folder, exist_ok=True)  # Create the directory if it doesn't exist

# Create the files in the respective directories
for folder, files in files_to_create.items():
    for file_name in files:
        file_path = os.path.join(folder, file_name)
        with open(file_path, 'w') as f:
            # Optionally, you can write some initial content to the files
            if file_name == "__init__.py":
                f.write("# Initialization file for package\n")
            else:
                f.write("# This is the implementation of the {} agent\n".format(file_name.replace(".py", "")))

print("Folder structure and files created successfully.")
