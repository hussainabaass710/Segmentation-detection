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


## License

MIT

