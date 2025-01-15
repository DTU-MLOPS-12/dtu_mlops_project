# MLOps-Driven Fine-Tuning of MobileNet-V4 for Vehicle Classification Using ImageNet

This repository is for the project in course 02476 - Machine Learning Operations at DTU with [the following structure.](#project-structure)

This project focuses on fine-tuning a MobileNet-V4 model, sourced from PyTorch Image Models (TIMM) utilizing the ImageNet dataset [described in the Project Description.](#project-description)  

## Project Description
Our new customer requires a custom and flexible image classification solution for real-world photos, specifically designed for robotics and edge AI platforms. A crucial requirement is the ability to rapidly add new classes while maintaining strict constraints on model size to ensure optimal inference performance on limited hardware resources. 

To secure this contract and demonstrate our expertise, we will develop an automated MLOps pipeline utilizing skills from the 02476 MLOps course and leveraging our combined team capabilities. 

The customer requests that we develop a proof-of-concept solution capable of training and deploying a model that accurately classifies real-world photos of trucks, buses, and cars. Subsequently, the model should expand to incorporate classes for motorcycles and bicycles.

For this challenge, we will select a pretrained model from PyTorch Image Models (TIMM) and fine-tune it using relevant classes from a subset of the ImageNet dataset. With over 10,000 classes available in ImageNet, our solution can adapt dynamically to meet the customer's evolving classification needs. 

We will begin by utilizing MobileNet-V4 in various sizes like `mobilenetv4_conv_small` and `mobilenetv4_hybrid_medium`, which has not been trained on the ImageNet dataset, to evaluate its performance and ensure high classification accuracy. 

To support our customers' need for various models and minimize boilerplate in our solution, we will get inspiration from the official training script from TIMM, a open-source third-party project. This approach will enable us to fine-tune a model using multiple hyperparameters and to leverage a vast number of TIMM classification models in our solution.

Subsequently, we plan to optimize the model through quantization techniques and create a robust classification endpoint utilizing FastAPI. Docker is leveraged to provide system-level reproducibility combined with efficient deployment workflows using Continuous Integration (CI) and monitoring, ensuring customer satisfaction.

Resources:
- [PyTorch Image Models (TIMM) Documentation](https://github.com/huggingface/pytorch-image-models?tab=readme-ov-file#getting-started-documentation)
- [MobileNet-V4](https://huggingface.co/blog/rwightman/mobilenetv4)
- [Dataset imagenet-1k-wds (subset)](https://huggingface.co/datasets/timm/imagenet-1k-wds)


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

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template)


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


### Dataset

- Download validation dataset
```bash
docker run --rm --name experiment1 -e HUGGING_FACE_HUB_TOKEN=add_token -v $(pwd)/data/processed:/data/processed/ -v $(pwd)/reports:/reports data:latest process-splits --splits validation --buffer-size 1000 --batch-size 1000
```

- Download train dataset
```bash
docker run --rm --name experiment1 -e HUGGING_FACE_HUB_TOKEN=add_token -v $(pwd)/data/processed:/data/processed/ -v $(pwd)/reports:/reports data:latest process-splits --splits train --buffer-size 10000 --batch-size 1000
```

- Test run using timm
```bash
python src/dtu_mlops_project/timm/train.py --config src/dtu_mlops_project/mobilenetv4_conv_small_test.yaml --data-dir data/processed/timm-imagenet-1k-wds-subset --dataset ImageFolder --train-split train --val-split validation --device cpu --num-classes 3 --epochs 1
```


### GCP

- Login
```bash
gcloud auth login
gcloud auth application-default login
```

- Projects list
```bash
gcloud projects list
```

- Config project with dtu-mlops PROJECT_ID
```bash
gcloud config set project <project-id>
```



### DVC Setup with Public Remote Storage

- Initialized DVC repository.
```bash
dvc init
```

- Add remote storage and support for object versioning
```bash
dvc remote add -d remote_storage gs://mlops_grp_12_data_bucket_public/
git add .dvc/config
```

- Support for object versioning
```bash
dvc remote modify remote_storage version_aware true
```

- Remove Public action prevention from the bucket 
```bash
gcloud storage buckets update gs://mlops_grp_12_data_bucket_public/ --no-public-access-prevention
```

- Make the bucket public using the below command
```bash
gcloud storage buckets add-iam-policy-binding gs://mlops_grp_12_data_bucket_public/ --member=allUsers --role=roles/storage.objectViewer
```

- Verify login 
```bash
gcloud auth application-default login
```

- Add processed data
```bash
dvc add data/processed/timm-imagenet-1k-wds-subset/
git add data/processed/timm-imagenet-1k-wds-subset.dvc
git add data/processed/.gitignore
git commit -m "First datasets"
git tag -a "v1.0" -m "data v1.0"
dvc push
```

## Building and pushing an image to the artifact registry

To build an OCI image using `docker`, simply run the following from the root directory of the project:

```sh
docker build -f dockerfiles/<service-name>.dockerfile . -t <tag-name>:<version-name>
```
where `<service-name>` is the name of the service container to build (`api`, `train`, etc.),
`<tag-name>` is the name of the tag and the `<version-name>` is the name of the version 
of the image, i.e. `latest`, `dev` `1.0.0`. While the `<service-name>` and the `<tag-name>`
can be different, it is easier if they are the same, so one can tell which dockerfile is 
used to create a given image.

Now, we create a tag and upload it to our artifact registry on GCP:

```sh
docker tag <tag-name> europe-docker.pkg.dev/dtu-mlops-447711/default-container-repository/<tag-name>:<version-name>
docker push europe-docker.pkg.dev/dtu-mlops-447711/default-container-repository/<tag-name>:<version-name>
```

### Make life easier with `docker compose`

You can also use `docker compose` to build and run multiple containers at once.
From the project root directory, simply run:

```shell
docker compose up -d
```

Now, `docker` will build or fetch all images required to run the specified services and
subsequently start them.
