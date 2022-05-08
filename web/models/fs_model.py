import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable

from .base_model import BaseModel
from . import networks
from .fs_networks import Generator_Adain_Upsample, Discriminator


class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size-1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))
        return return_images


class SpecificNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(SpecificNorm, self).__init__()
        self.mean = np.array([0.485, 0.456, 0.406])
        self.mean = torch.from_numpy(self.mean).float().cpu()
        self.mean = self.mean.view([1, 3, 1, 1])
        self.std = np.array([0.229, 0.224, 0.225])
        self.std = torch.from_numpy(self.std).float().cpu()
        self.std = self.std.view([1, 3, 1, 1])

    def forward(self, x):
        mean = self.mean.expand([1, 3, x.shape[2], x.shape[3]])
        std = self.std.expand([1, 3, x.shape[2], x.shape[3]])
        return (x - mean) / std

    
class fsModel(BaseModel):
    def name(self):
        return 'fsModel'

    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss):
        flags = (True, use_gan_feat_loss, use_vgg_loss, True, True, True, True, True)

        def loss_filter(g_gan, g_gan_feat, g_vgg, g_id, g_rec, g_mask, d_real, d_fake):
            return [l for (l, f) in zip((g_gan, g_gan_feat, g_vgg, g_id, g_rec, g_mask, d_real, d_fake), flags) if f]

        return loss_filter

    def initialize(self, home_dir):
        BaseModel.initialize(self, home_dir)
        
        self.isTrain = False
        device = torch.device("cpu")

        # Generator networkload
        self.netG = Generator_Adain_Upsample(input_nc=3, output_nc=3, latent_size=512, n_blocks=9, deep=False).to(device)

        # Id network
        self.netArc = torch.load(os.path.join(home_dir, "weights/arcface.tar"), map_location=torch.device('cpu'))['model'].module.to(device)
        self.netArc.eval()

        self.load_network(self.netG, 'G', "latest", "")
        return


    def forward(self, img_id, img_att, latent_id, latent_att, for_G=False):
        return self.netG.forward(img_att, latent_id)
        


