#!/usr/bin/env python3
"""
Setup script for Amazon Reviews Sentiment Analysis project
"""

import os
import subprocess
import sys

def run_command(command, description):
    """Run a shell command and print status"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} failed with error: {e}")
        return False

def create_directories():
    """Create necessary project directories"""
    directories = [
        'data',
        'models', 
        'notebooks',
        'results',
        'logs'
    ]
    
    print("📁 Creating project directories...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✓ Created/verified: {directory}/")

def install_requirements():
    """Install Python requirements"""
    if os.path.exists('requirements.txt'):
        return run_command(f"{sys.executable} -m pip install -r requirements.txt", 
                          "Installing Python packages")
    else:
        print("❌ requirements.txt not found!")
        return False

def download_nltk_data():
    """Download required NLTK datasets"""
    nltk_downloads = [
        'punkt',
        'stopwords',
        'punkt_tab'
    ]
    
    print("📚 Downloading NLTK data...")
    try:
        import nltk
        for dataset in nltk_downloads:
            try:
                nltk.download(dataset, quiet=True)
                print(f"  ✓ Downloaded: {dataset}")
            except Exception as e:
                print(f"  ⚠️  Warning: Could not download {dataset}: {e}")
        return True
    except ImportError:
        print("❌ NLTK not installed. Install requirements first.")
        return False

def create_sample_config():
    """Create a sample configuration file"""
    config_content = """# Amazon Reviews Sentiment Analysis - Configuration

# Data paths
DATA_PATH = "data/1429_1.csv"
TRAIN_DATA_PATH = "data/train.csv"
TEST_DATA_PATH = "data/test.csv"

# Model paths
MODEL_PATH = "models/naive_bayes_model.pkl"
METRICS_PATH = "models/metrics.pkl"

# Training parameters
TEST_SIZE = 0.3
RANDOM_STATE = 42
BALANCE_DATASET = True

# Text preprocessing
MIN_WORD_LENGTH = 2
REMOVE_STOPWORDS = True
USE_STEMMING = True

# Evaluation
GENERATE_PLOTS = True
SAVE_RESULTS = True
"""
    
    print("⚙️  Creating sample configuration...")
    try:
        with open('config.py', 'w') as f:
            f.write(config_content)
        print("✓ Configuration file created: config.py")
        return True
    except Exception as e:
        print(f"❌ Failed to create config: {e}")
        return False

def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# celery beat schedule file
celerybeat-schedule

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# Project specific
data/*.csv
!data/sample_*.csv
models/*.pkl
models/*.joblib
results/
logs/
*.log

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
"""
    
    print("📝 Creating .gitignore...")
    try:
        with open('.gitignore', 'w') as f:
            f.write(gitignore_content)
        print("✓ .gitignore file created")
        return True
    except Exception as e:
        print(f"❌ Failed to create .gitignore: {e}")
        return False

def verify_installation():
    """Verify that the installation was successful"""
    print("\n🔍 Verifying installation...")
    
    # Check if required files exist
    required_files = [
        'train_model.py',
        'predict.py', 
        'requirements.txt',
        'config.py'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✓ Found: {file}")
        else:
            print(f"  ❌ Missing: {file}")
            missing_files.append(file)
    
    # Check if directories exist
    required_dirs = ['data', 'models', 'notebooks', 'results', 'logs']
    for directory