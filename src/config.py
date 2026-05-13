from pathlib import Path

# Native WSL paths
PYTHON_PROJECT_DIR = Path("/home/horia/assertion-generator-java")
DATA_PROJECT_DIR = Path("/home/horia/llm-prompts-empirical-study")

# This must match where we actually ran 'git clone'
CLONED_REPOS_DIR = DATA_PROJECT_DIR / "input" / "github-repos"

# Explicitly define the dataset path found by your search
DATASET_PATH = DATA_PROJECT_DIR / "scripts" / "test-miner" / "output" / "miner" / "twilio" / "twilio-java.json"

# Other paths
PROMPTS_DIR = DATA_PROJECT_DIR / "scripts" / "dataset" / "output" / "raw-oracles-dataset"
SANDBOX_DIR = PYTHON_PROJECT_DIR / "sandbox"
MUTATION_JAR_PATH = DATA_PROJECT_DIR / "scripts" / "dataset" / "target" / "dataset-jar-with-dependencies.jar"
JAVA_HOME = Path("/usr/lib/jvm/java-21-openjdk-amd64")
ORACLE_CONFIG_JSON = DATA_PROJECT_DIR / "scripts" / "dataset" / "src" / "main" / "resources" / "oracles-dataset_config.json"