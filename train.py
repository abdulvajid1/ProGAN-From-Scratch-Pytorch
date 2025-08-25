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
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.Normalize(
                [0.5 for _ in range(len(config.CHANNEL_IMG))],
                [0.5 for _ in range(len(config.CHANNEL_IMG))]
            )
        ]
    )
    
    batch_size = config.BATCH_SIZE[int(log2(image_size/4))]
    dataset = datasets.ImageFolder(root=config.DATASET, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    return loader, dataset

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
    
    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(config.DEVICE)
        curr_batch_size = curr_batch_size.shape[0]
        
        # Train critic: min(- (critic(real) - critic(fake)))
        
        noice = torch.randn(curr_batch_size, config.Z_DIM, 1, 1).to(config.DEVICE)
        
        with torch.cuda.amp.autocast():
            fake = generator(noice, alpha, step)
            critic_real = critic(real, alpha, step)
            critic_fake = critic(fake.detach(), alpha, step)
            
        gp = gradient_penalty(critic, real, fake, alpha, step, device=config.DEVICE)
        
        loss_critic = - (torch.mean(critic_real) - torch.mean(critic_fake)) + gp * config.LAMBDA_GP + (0.001 * torch.mean(critic_real**2))
        
        opt_critic.zero_grad()
        scalar_critic.scale(loss_critic).backward()
        scalar_critic.step(opt_critic)
        scalar_critic.update()
        
        # Train Generator
        
        with torch.cuda.amp.autocast():
            critic_fake = critic(fake, alpha, step)
            loss_gen = - torch.mean(critic_fake)
            
        opt_gen.zero_grad()
        scalar_gen.scale(loss_gen).backward()
        scalar_gen.step(opt_gen)
        scalar_gen.update()
        
        alpha += curr_batch_size / (len(dataset) * config.PROGRESSIVE_EPOCH[step] * 0.5)
        alpha = min(alpha, 1) # alpha should not go above 1
              

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
        
        loader , dataset = get_loader(4*2**step) # 4*2^0 = 4, so initially start from 4 then 8, 16
        
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