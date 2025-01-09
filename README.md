# dtu_mlops_project
TODO: Give the project a title^

02476 Machine Learning Operations Project at DTU

## Project Description
Our new customer requires a custom and flexible image classification solution for real-world photos, specifically designed for robotics and edge AI platforms. A crucial requirement is the ability to rapidly add new classes while maintaining strict constraints on model size to ensure optimal inference performance on limited hardware resources. 

To secure this contract and demonstrate our expertise, we will develop an automated MLOps solution utilizing skills from the 02476 MLOps course and leveraging our combined team capabilities. 

The customer demands that we create a proof-of-concept system capable of training and deploying a model that accurately classifies real-world photos of trucks, buses, cars, and efficiently fine-tunes the model to also incorporate classes for motorcycles and bicycles. 

For this challenge, we will select a pretrained model from PyTorch Image Models (TIMM) and fine-tune it using relevant classes from a subset of the ImageNet dataset. With over 10,000 classes available in ImageNet, our solution can adapt dynamically to meet the customer's evolving classification needs. 

We will begin by utilizing MobileNet-V4 in various sizes, which has not been trained on the ImageNet dataset, to evaluate its performance and ensure high classification accuracy. To support our customers' need for various models and minimize boilerplate in our solution, we will incorporate the official training script from TIMM. This approach will enable us to fine-tune the model using multiple hyperparameters and to leverage a vast number of TIMM classification models in our solution.

Subsequently, we plan to optimize the model through quantization techniques and create a robust classification endpoint utilizing FastAPI.

## Project Structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).




## MLops tools

### Dev Environment Setup
- Setup first time
```bash
cd /dtu_mlops_project 
sudo apt install python3.12-venv # Optional if not installed
python3 -m venv env  # create a virtual environment in project
source env/bin/activate  # activate that virtual environment
```

- Install pip requirements
```bash
# install requirements.txt
pip install .
# or in developer mode requirements_dev.txt
pip install -e .
```

- Open project in VS Code with active env
```bash
code .
```

- Deactivate env and delete files (debugging)
```bash
cd /dtu_mlops_project
deactivate
python3 -m venv --clear env
```
### Linting

This [page](https://docs.astral.sh/ruff/) contains an overview of the Ruff tool, which is an extremely fast code linter and formatter. As an example, you can run the following:

```bash
ruff check tasks.py
```

### Typing
This [cheat sheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html) is a good resource on typing. Run [mypy](https://mypy.readthedocs.io/en/stable/index.html) on the `tasks.py` file
```bash
mypy tasks.py
```

## Invoke Commands
Information about all available executable tasks:
 ```bash
 invoke --list
 ```

### Setup Commands
- Create a new conda environment for the project:
    ```bash
    invoke create-environment
    ```

- Install project requirements from requirements.txt:
    ```bash
    invoke requirements
    ```

- Install development requirements from requirements_dev.txt:
    ```bash
    invoke dev-requirements
    ```

### MLOps Commands
- Execute the data preprocessing pipeline, which includes data cleaning, normalization, and feature engineering:
    ```bash
    invoke preprocess-data
    ```

- Train the machine learning model using the preprocessed data and specified configuration settings:
    ```bash
    invoke train
    ```

- Run the test suite to evaluate the performance and accuracy of the trained model:
    ```bash
    invoke test
    ```

- Build Docker images for the project, including environments for training and API deployment:
    ```bash
    invoke docker-build
    ```

- Run and serve the backend API
    ```bash
    invoke runserver
    ```

### Documentation Commands
-  Build documentation from the docs directory into a static website:
    ```bash 
    invoke build-docs
    ```

- Starts a local server that can be used to view the documentation from the docs directory:
    ```bash
    invoke serve-docs
    ```

## CLI's

This project comes with a few different scripts, each with their own options:

### `data.py`

### `evaluate.py`

### `model.py`

### `train.py`

### `visualize.py`
