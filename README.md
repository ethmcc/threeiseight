# Proof that 8 is just one bite away from 3

This project demonstrates how a neural network perceives a partially obscured digit 8 as the digit 3.

## Requirements

- Python 3.9
- Dependencies listed in `requirements.txt`

## Setup

1. Clone this repository
2. Run the setup script:

```bash
sh setup.sh
```

The script will:
- Create a virtual environment
- Install dependencies
- Train the digit recognition model (if needed)

## Usage

Activate the virtual environment and run the main script:

```bash
source venv/bin/activate
python main.py
```

This will:
1. Generate images of digits 0-9
2. Test that the model correctly identifies each digit
3. Demonstrate how a "bitten" 8 is recognized as 3

## Output

Generated images will be saved in the `out` directory:
- `original_8.png` - The original digit 8
- `bitten_8.png` - The digit 8 after being "bitten"

The model will confirm that the bitten 8 is recognized as 3, proving our hypothesis.

