import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2

factors = [1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]


class WsConv2d(nn.Module):
    "Weigted scale (equlized learning rate)"
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (gain/ (in_channels * kernel_size**2)) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None
        
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)
        
    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)
        
        
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.epilson = 1e-8
        
    def forward(self, x):
        return x / torch.sqrt((torch.mean(x**2, dim=1, keepdim=True) + self.epilson))
    
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm=True):
        super().__init__()
        self.conv1 = WsConv2d(in_channels, out_channels)
        self.conv2 = WsConv2d(out_channels, out_channels)
        self.pixel_norm = PixelNorm()
        self.leaky = nn.LeakyReLU(0.2)
        self.use_norm = use_norm

    def forward(self, x):
        x = self.leaky(self.conv1(x))
        if self.use_norm:
            x = self.pixel_norm(x)
        
        x = self.leaky(self.conv2(x))
        if self.use_norm:
            x = self.pixel_norm(x)
        
        return x

class Generator(nn.Module):
    def __init__(self, z_dim, in_channels, img_channels=3):
        super().__init__()
        self.z_dim = z_dim
        
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(z_dim, in_channels, 4, 1, 0),
            nn.LeakyReLU(),
            PixelNorm(),
            WsConv2d(in_channels, in_channels),
            nn.LeakyReLU(),
            PixelNorm()
        )
        
        self.initial_rgb = WsConv2d(in_channels, img_channels, kernel_size=1, stride=1, padding=0)
        
        self.progress_block, self.rgb_layers = nn.ModuleList(), nn.ModuleList([self.initial_rgb])
        
        
        for i in range(len(factors) -1):
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i+1])
            
            self.progress_block.append(ConvBlock(conv_in_c, conv_out_c))
            self.rgb_layers.append(WsConv2d(conv_out_c, img_channels, kernel_size=1, stride=1, padding=0))
            
        
    def fade_in(self, alpha, upscaled, generated):
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)
    
    def forward(self, x, steps, alpha): # if step=0 (4,4) if step=1 (8,8) progess..
        
        out = self.initial(x) # (b, 512, 1, 1) -> (b, 512, 4, 4)
        
        if steps == 0:
            return self.initial_rgb(out)
        
        for step in range(steps):
            upscaled = F.interpolate(out, scale_factor=2, mode='nearest') 
            out = self.progress_block[step](upscaled)
            
        final_upscaled = self.rgb_layers[steps-1](upscaled)
        final_out = self.rgb_layers[steps](out)
        
        return self.fade_in(alpha, final_upscaled, final_out)
             
class Discriminator(nn.Module):
    def __init__(self, in_channels, img_channels=3):
        super().__init__()
        self.prgress_block, self.rgb_layers = nn.ModuleList(), nn.ModuleList()
        self.leaky = nn.LeakyReLU(0.2)
        
        for i in range(len(factors) - 1, 0, -1):
            c_in_channel = int(in_channels * factors[i])
            c_out_channel = int(in_channels * factors[i-1])
            self.prgress_block.append(ConvBlock(c_in_channel, c_out_channel, use_norm=False))
            self.rgb_layers.append(WsConv2d(img_channels, c_in_channel, kernel_size=1, stride=1, padding=0))
            
        self.last_layer = nn.Sequential(
            WsConv2d(in_channels+1, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            WsConv2d(in_channels, in_channels, kernel_size=4, padding=0, stride=1),
            nn.LeakyReLU(0.2),
            WsConv2d(in_channels, 1, kernel_size=1, padding=0, stride=1),
        )
        
        self.initial_rgb = WsConv2d(img_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.rgb_layers.append(self.initial_rgb)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
    def fade_in(self, alpha, downscaled, out):
        return alpha * out + (1-alpha) * downscaled
    
    def minibatch_std(self, x):
        batch_statistics_map = torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        return torch.concat([x, batch_statistics_map], dim=1)
        
    def forward(self, x, alpha, steps):
        curr_steps = len(self.prgress_block) - steps
        out = self.leaky(self.rgb_layers[curr_steps](x))
        
        if steps == 0:
            out = self.minibatch_std(out)
            return self.last_layer(out).view(out.shape[0], -1)
        
        downscaled = self.leaky(self.rgb_layers[curr_steps + 1](self.avg_pool(x)))
        out = self.avg_pool(self.prgress_block[curr_steps](out)) 
        
               
        out = self.fade_in(alpha, downscaled, out)
        
        for step in range(curr_steps + 1, len(self.prgress_block)):
            out = self.prgress_block[step](out)
            out = self.avg_pool(out)
            
        out = self.minibatch_std(out)
        
        return self.last_layer(out).view(out.shape[0], -1)
    
    
if __name__ == "__main__":
    z_dim = 50
    in_channels = 256
    gen = Generator(z_dim, in_channels, img_channels=3)
    critic = Discriminator(in_channels, img_channels=3)
    
    for image_size in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        num_steps = int(log2(image_size/4))
        x = torch.randn((1, z_dim, 1, 1))
        z = gen(x, num_steps, 0.5)
        assert z.shape == (1, 3, image_size, image_size)
        
        out = critic(z, 0.5, num_steps)
        
        assert out.shape == (1, 1)
        
        print(f'success {image_size}')
        