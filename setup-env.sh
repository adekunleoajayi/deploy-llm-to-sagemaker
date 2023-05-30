#!/bin/bash

python3 -m venv venv

source venv/bin/activate

python -m pip install --upgrade pip

pip install -U setuptools

if [ -f "requirements.txt" ]; then
    echo "requirements.txt file found. Installing packages..."
    pip install -r requirements.txt
else
    echo "requirements.txt file not found."
fi

if [ -f "requirements-dev.txt" ]; then
    echo "requirements-dev.txt file found. Installing packages..."
    pip install -r requirements-dev.txt
else
    echo "requirements-dev.txt file not found."
fi

echo " Your environment is ready and can be activated by running: source venv/bin/activate "
