# PyTorch-Demo


## Project Overview
A simple CNN and a ResNet-18 comparison in PyTorch for image recognition:
- Data loading & preprocessing for MNIST and CIFAR-10  
- Reusable training & evaluation code  
- Hyperparameter sweeps and visualization in Jupyter 
- Unit tests for data loaders and models 
- Checkpoint tracking

## How to install

```bash
git clone https://github.com/IgorSuvalov/PyTorch‑Demo.git
cd PyTorch-Demo
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Project Structure
```bash
-src/              # Data loader, CNN model, ResNet model, training
-tests/            # unit tests
-models/           # Best model checkpoints
-data/             # MNIST and CIFAR-10 datasets
-notebooks/        # Model overview and visualization
-train.py          # Train function with parsers
-README.md         # Project overview (this file)
-requirements.txt  # Python packages
```

## Training with CLI
```bash
# SimpleCNN on MNIST
python train.py --dataset mnist --model simple --batch_size 64 --epochs 10 --lr 1e-3 --data_dir data --model_dir models/simple_mnist 

# ResNet on CIFAR-10
python train.py --dataset cifar10 --model resnet --batch_size 128 --epochs 20 --lr 0.01 --data_dir data --model_dir models/resnet_cifar --momentum 0.9
```

## Testing
```bash
pytest --maxfail=1 --disable-warnings -q
```

## Results
Best **test accuracy** on MNIST for both CNN and ResNet ~99%.

Best **test accuracy** on CIFAR-10 for ResNet ~74% and for CNN ~72%

## License
MIT

