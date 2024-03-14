#!/bin/bash

#SBATCH --mail-user=t.witter@campus.lmu.de
#SBATCH --mail-type=ALL
#SBATCH --partition=All
#SBATCH --export=NONE

# Environment Variables
# export WANDB_SILENT="true" # Use if you want to not disable wandb logging
export WANDB_MODE="disabled"

#export PYENV_ROOT="$HOME/.pyenv"
#command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
#eval "$(pyenv init -)"

# pyenv setup
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
export PATH="$PYENV_ROOT/shims:${PATH}"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# Check if pyenv is installed
if command -v pyenv 1>/dev/null 2>&1; then
    # Setup pyenv shell
    eval "$(pyenv init --path)"
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"

    # Create a fresh virtual environment
    pyenv virtualenv env 1>/dev/null 2>&1
    pyenv activate env 1>/dev/null 2>&1

    # Check the exit status of the pyenv activate command
    if [ $? -ne 0 ]; then
        echo "\033[31mFailed to activate the virtual environment using pyenv. Exiting.\033[0m"
        exit 1
    fi

# Check if virtualenv is installed
elif command -v virtualenv 1>/dev/null 2>&1; then
    # Create a fresh virtual environment using virtualenv
    virtualenv env
    source env/bin/activate

    # Check the exit status of the virtual environment activation
    if [ $? -ne 0 ]; then
        echo "\033[31mFailed to activate the virtual environment using virtualenv. Exiting.\033[0m"
        exit 1
    fi
# If neither is installed, quit!
else
    echo "\033[31mNeither pyenv nor virtualenv are available. Exiting.\033[0m"
    exit 1
fi

# Makes sure that newly added modules are installed aswell
pip install -qr requirements.txt

# Runs the script
python src/main.py $@