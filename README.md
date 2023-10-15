# AutoEye

This project was ddeveloped during Datasaur hachathon.

We have provided a solution to the `case 3`. The challenge of `case 3` is to predict wether an image of a car is valid (was taken from repairment service place).

## Data

The dataset and challenge is available at Kaggle [here](https://www.kaggle.com/competitions/case3-datasaur-photo/overview)

## Project

Our project consists of 3 parts:

- Neural Network (PyTorch)
- Backend (FastAPI)
- Telegram Bot (Python)

## Model Architecture

Our model consists of 2 parts:

- Meta's Dino v2
- SVM

Dino v2 was pretrained and converts the given image to the embeddings, which are then fed to the SVM.
The output of our model is a class from:

- 0: _*Correct*_
- 1: _*Not-on-the-brake-stand*_
- 2: _*From-the-screen*_
- 3: _*From-the-screen+photoshop*_
- 4: _*Photoshop*_

## Scripts

Run all the scripts from the root directory of the repo

creates a csv file that is used during training. Make sure toextract the dataset archive into data folder:

```sh
python ./scripts/create-csv.py train
```

start a server on port 8000:

```sh
./scripts/start_server.sh
```

## Usage

### Dependencies

To use this project you must create a venv:

```sh
python3 -m venv venv
```

Activate the environment:

```sh
source ./venv/bin/activate
```

Install requirements:

```sh
pip install -r requirements.txt
```

### Download the files

- Create directory `/models` at the root of the project:

```sh
mkdir models
```

- Download `classifier.pickle` from releases [here](https://github.com/CommanderXA/AutoEye/releases/tag/multiclass)

- Put this `classifier.pickle` to the `/models` directory

### Inference

To generate test predictions:

```sh
python nn/main.py
```

To validate model with sample from train dataset:

```sh
python nn/evaluate.py
```

To train the model:

```sh
python nn/train.py
```

### Configuration

It is possible to tweak some of the parameters of the NN including the base model itself (possible to swap between).
Config file is located at `/nn/conf/config.yaml`.

You need to generate models in `main.ipynb`.
Models options:

- `dino_vision` - the best model (available at releases)
- `dino`
- `resnet`

## Bot

The Telegram bot is available at this [link](https://t.me/AutoEyeBot)
