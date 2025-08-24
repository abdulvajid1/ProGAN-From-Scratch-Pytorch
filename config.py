from math import log2
import torch

START_TRAIN_AT_IMAGE_SIZE = 4
DATASET = ''
CHECKPOINT_GEN = 'generator.ckpt'
CHECKPOINT_DISC = 'discriminator.ckpt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_MODEL = True
LOAD_MODEL = False
LR = 2e-3
BATCH_SIZE = [16, 16, 16, 16, 16, 16, 16, 8, 4]
IMAGE_SIZE = 1024
CHANNEL_IMG = 3
Z_DIM = 256
IN_CHANNELS = 256
LAMBDA_GP = 10
NUM_STEPS = int(log2(IMAGE_SIZE/ 4)) + 1
PROGRESSIVE_EPOCH = [20] * len(BATCH_SIZE)
FIXED_NOICE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE)
NUM_WORKERS = 1