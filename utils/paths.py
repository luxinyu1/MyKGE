from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent

DATASETS_DIR = REPO_DIR / 'datasets'
DATASETSMULTI_DIR = REPO_DIR / 'datasets_multi'
DATASETPLAIN_DIR = REPO_DIR / 'datasets_plain'

CHECKPOINTS_DIR = REPO_DIR / 'checkpoints'

LOG_DIR = REPO_DIR / 'logs'
TENSORBOARD_DIR = LOG_DIR / 'tensorboard'

def get_dataset_dir(dataset):
    return DATASETS_DIR / dataset

def get_dataset_multi_dir(dataset):
    return DATASETSMULTI_DIR / dataset

def get_dataset_plain_dir(dataset):
    return DATASETPLAIN_DIR / dataset
