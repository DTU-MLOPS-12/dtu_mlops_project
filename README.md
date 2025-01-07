# dtu_mlops_project

02476 Machine Learning Operations Project

## Authors
TODO

## Project Description
TODO 

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

## Relevant Commands

## Invoke Commands
- `invoke --list`: Information about all available tasks

### Setup Commands
- `invoke create-environment`: Create a new conda environment for the project.
- `invoke requirements`: Install project requirements from requirements.txt.
- `invoke dev-requirements`: Install development requirements from requirements.txt.

### MLOps Commands
- `invoke preprocess-data`: Execute the data preprocessing pipeline, which includes data cleaning, normalization, and feature engineering.
- `invoke train`: Train the machine learning model using the preprocessed data and specified configuration settings.
- `invoke test`: Run the test suite to evaluate the performance and accuracy of the trained model.
- `invoke docker-build`: Build Docker images for the project, including environments for training and API deployment.

### Documentation Commands
- `invoke build-docs`: Build documentation from the docs directory into a static website.
- `invoke serve-docs`: Starts a local server that can be used to view the documentation from the docs directory.

## TODO: Add more commands