import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import (
    gradient_penalty,
    plot_to_tensorboard,
    save_checkpoint,
    load_checkpoint,
    generate_examples
)
from model import Generator, Discriminator
from math import log2
from tqdm import tqdm
import config


torch.backends.cudnn.benchmark = True

def get_loader(image_size):
    pass

def train_fn(critic,
            generator,
            loader,
            dataset,
            step,
            alpha,
            opt_critic,
            opt_gen,
            tensorboard_step,
            writer,
            scalar_gen,
            scalar_critic
    ):
    loop = tqdm(loader, leave=True)
    

def main():
    generator = Generator(config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNEL_IMG).to(config.DEVICE)
    critic = Discriminator(config.IN_CHANNELS, img_channels=config.CHANNEL_IMG).to(config.DEVICE)
    
    opt_gen = optim.Adam(generator.parameters(), lr=config.LR, betas=(0.0, 0.999))
    opt_critic = optim.Adam(critic.parameters(), lr=config.LR, betas=(0.0, 0.999))
    
    scalar_critic = torch.amp.GradScaler(config.DEVICE)
    scalar_gen = torch.amp.GradScaler(config.DEVICE)
    
    writer = SummaryWriter(f'log/gan')
    
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, generator, opt_gen, config.LR)
        load_checkpoint(config.CHECKPOINT_DISC, critic, opt_critic, config.LR)
        
    generator.train()
    critic.train()
    
    tensorboard_step = 0
    step = int(log2(config.START_TRAIN_AT_IMAGE_SIZE / 4))
    
    for num_epochs in config.PROGRESSIVE_EPOCH[step: ]:
        alpha = 1e-5
        
        loader , dataset = get_loader(4*2**step)
        
        for epoch in range(num_epochs):
            tensorboard_step, alpha = train_fn(
                critic,
                generator,
                loader,
                dataset,
                step,
                alpha,
                opt_critic,
                opt_gen,
                tensorboard_step,
                writer,
                scalar_gen,
                scalar_critic
            )
            
            if config.SAVE_MODEL:
                save_checkpoint(generator, opt_gen, filename=config.CHECKPOINT_GEN)
                save_checkpoint(critic, opt_critic, filename=config.CHECKPOINT_DISC)    
                
        step +=1
    
    
    
if __name__ == '__main__':
    main()