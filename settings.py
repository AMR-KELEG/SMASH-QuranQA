import torch

GPU_ID = torch.device("cuda:3")
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 16
EPOCHS = 16
CROSS_ENTROPY_IGNORE_INDEX = -100
MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-ca"
