# Pointcloud Segmentation Stub

A stub project for pointcloud segmentation using deep learning.

## Project Structure

```
pointcloud-segmentation-stub/
│
├─ data/
│   └─ sample_pointclouds.npz   (synthetic or placeholder)
│
├─ pointcloud_segmentation/
│   ├─ __init__.py
│   ├─ dataset.py
│   ├─ model.py
│   ├─ train.py
│   ├─ utils.py
│
├─ tests/
│   ├─ test_dataset.py
│   └─ test_model.py
│
├─ pyproject.toml
├─ poetry.lock
├─ .gitignore
└─ README.md
```

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

### Prerequisites

- **Python 3.8.1 or higher** (Python 3.8.1+ required)
  - Check your Python version: `python3 --version`
  - The project requires Python >= 3.8.1, < 4.0

- **Poetry** - Install Poetry if you haven't already:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### Install Dependencies

```bash
poetry install
```

This will create a virtual environment and install all dependencies including:
- `torch` - PyTorch for deep learning
- `numpy` - Numerical computing
- `pytest` - Testing framework (dev dependency)
- `black`, `flake8`, `mypy` - Code quality tools (dev dependencies)

### Activate the Virtual Environment

There are several ways to use the Poetry virtual environment:

**Option 1: Use `poetry env activate` (Poetry 2.0+ recommended)**
```bash
cd ~/Documents/pointcloud-segmentation-stub
poetry env activate
```
This activates the virtual environment. Note: This is not a direct replacement for the old `shell` command, but works similarly.

**Option 2: Install the shell plugin (if you prefer the old `poetry shell` command)**
```bash
poetry self add poetry-plugin-shell
```
Then you can use:
```bash
poetry shell
```
To deactivate, type `exit` or press `Ctrl+D`.

**Option 3: Run commands directly (no activation needed)**
```bash
poetry run python <script>
poetry run pytest tests/
poetry run python create_sample_data.py
```
This runs commands in the virtual environment without activating it. Useful for one-off commands or scripts.

**Option 4: Manual activation (using the environment path)**
```bash
source $(poetry env info --path)/bin/activate
```
This manually activates the virtual environment using the standard Python venv activation method.

**Find the virtual environment path:**
```bash
poetry env info --path
```

**List all Poetry environments:**
```bash
poetry env list
```

## Usage

### Training

```python
from pointcloud_segmentation.train import train

config = {
    'train_data_path': 'data/sample_pointclouds.npz',
    'val_data_path': 'data/sample_pointclouds.npz',
    'batch_size': 32,
    'num_classes': 10,
    'num_points': 1024,
    'learning_rate': 0.001,
    'num_epochs': 10
}

train(config)
```

### Running Tests

Using Poetry:
```bash
poetry run pytest tests/
```

Or activate the environment first:
```bash
poetry shell
pytest tests/
```

### Generating Sample Data

To create the sample pointcloud data file:
```bash
poetry run python create_sample_data.py
```

## Data Format

The `.npz` file should contain:
- `pointclouds`: Array of shape (N, num_points, 3) containing pointcloud coordinates
- `labels`: Array of shape (N, num_points) containing segmentation labels

## Model Architecture

The model uses a simple 1D convolutional architecture for pointcloud segmentation:
- Input: (B, 3, N) where B is batch size and N is number of points
- Output: (B, num_classes, N) segmentation logits

## License

MIT

