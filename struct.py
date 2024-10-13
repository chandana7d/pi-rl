import os

# Define the folder structure as a dictionary
folder_structure = {
    "pi-library": {
        "README.md": "",
        "LICENSE": "",
        "setup.py": "",
        "pyproject.toml": "",
        "requirements.txt": "",
        ".gitignore": "",
        "docs": {
            "index.md": "",
            "api_reference.md": "",
            "architecture_overview.md": "",
            "tutorials": {
                "getting_started.md": ""
            }
        },
        "src": {
            "pi": {
                "__init__.py": "",
                "agents": {
                    "base_agent.py": "",
                    "dqn_agent.py": "",
                    "a3c_agent.py": ""
                },
                "experiences": {
                    "experience_buffer.py": ""
                },
                "policies": {
                    "dqn_policy.py": ""
                },
                "learners": {
                    "dqn_learner.py": ""
                },
                "models": {
                    "neural_networks.py": ""
                },
                "env": {
                    "base_env.py": ""
                },
                "train": {
                    "trainer.py": ""
                },
                "serve": {
                    "model_server.py": ""
                },
                "utils": {
                    "logging.py": ""
                },
                "distributed": {
                    "parameter_server.py": ""
                }
            },
            "tests": {
                "test_agents.py": "",
                "test_policies.py": "",
                "test_models.py": ""
            }
        },
        "examples": {
            "distributed_training.py": ""
        },
        "benchmarks": {
            "atari_benchmark.py": ""
        },
        "deployments": {
            "k8s": {
                "deployment.yaml": ""
            },
            "terraform": {
                "main.tf": ""
            }
        },
        "configs": {
            "agent_config.yaml": ""
        },
        "scripts": {
            "run_experiment.py": ""
        },
        "containers": {
            "Dockerfile": "",
            "docker-compose.yml": ""
        },
        "ci_cd": {
            "github_actions": {
                "ci_pipeline.yml": ""
            },
            "circleci": {
                "config.yml": ""
            }
        },
        "monitoring": {
            "dashboard": {
                "dashboard.py": ""
            },
            "debugger": {
                "distributed_debugger.py": ""
            }
        }
    }
}

# Function to create the folder structure
def create_structure(base_path, structure):
    for name, content in structure.items():
        # Create the directory or file
        path = os.path.join(base_path, name)
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)  # Create a directory
            create_structure(path, content)  # Recursively create subdirectories
        else:
            with open(path, 'w') as f:  # Create an empty file
                f.write(content)

# Specify the base path where the folder structure will be created
base_path = "pi-library"  # Change this to your desired base path

# Create the folder structure
create_structure(base_path, folder_structure)

print("Folder structure created successfully!")
