#!/bin/sh

# Check for Python 3.9
if ! command -v python3.9 &> /dev/null; then
    echo "Error: Python 3.9 is required but not found"
    echo "Please install Python 3.9 and try again"
    exit 1
fi

# Create and activate virtual environment with Python 3.9
python3.9 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Train the model if it doesn't exist
if [ ! -f "digit_model.keras" ]; then
    python train_model.py
fi

echo "Setup complete! To activate the environment, run: source venv/bin/activate"
