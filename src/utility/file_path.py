import os.path

DIRECTORY_PATH = os.getcwd()
REPO_PATH = DIRECTORY_PATH.split(os.sep)
REPO_PATH = os.sep.join(REPO_PATH[:-2])

LOG_PATH = os.path.join(REPO_PATH, 'log')
OBJECT_PATH = os.path.join(REPO_PATH, 'objectset')
DATASET_PATH = os.path.join(REPO_PATH, 'dataset')
RESULT_PATH = os.path.join(REPO_PATH, 'result')
MODEL_PATH = os.path.join(REPO_PATH, 'model')
