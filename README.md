# PyTorch-Demo


## Project Overwiev
A CNN in PyTorch for image recognition:
- Load and preprocess the MNIST dataset
- Train the CNN model
- Evaluate the model on the test set
- Use the unit tests to check the preprocessing and the model
- Plot the predictions of the best model

## How to install

```bash
git clone https://github.com/IgorSuvalov/PyTorch-Demo.git
cd pytorch-image-demo
python -m -venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

## How to use
-To train:
  python -m src.train
-To run tests:
  pytest -q

## Project Structure
scr/        # Data loader, model, training
tests/      # unit tests
models/     # Best model
data/       # MNIST dataset
notebooks/  # Model overview and visualization
README.md
requirements.txt

## Results
Best model's test accuracy is $~99%$ 
