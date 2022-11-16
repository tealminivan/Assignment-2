"""solver.py"""


import torch
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import os
from tqdm import tqdm
import visdom
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image

from utils import cuda, grid2gif
from model import BetaVAE
from dataset import return_data
from collections import deque

import matplotlib.pyplot as plt


def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
        # recon_loss = F.binary_cross_entropy_with_logits(x_recon, x)
    elif distribution == 'gaussian':
        x_recon = torch.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    else:
        recon_loss = None

    return recon_loss
    

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld
    

class Solver(object):
    def __init__(self, args):
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            print("**using GPU now**")
            self.device = torch.device('cuda', args.gpu)
        else:
            self.device = torch.device('cpu')
        
        self.max_iter = args.max_iter
        self.global_iter = 0

        self.z_dim = args.z_dim
        self.out_dim = args.out_dim
        self.beta = args.beta
        self.objective = args.objective
        self.model = args.model
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.load_ckp = args.load_ckp
        # self.dset_dir = args.dset_dir
        self.batch_size = args.batch_size
        self.save_step = args.save_step

        if args.dataset.lower() == 'mnist':
            self.nc = 1
            self.decoder_dist = 'bernoulli'
        elif args.dataset.lower() == 'svhn':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        elif args.dataset.lower() == 'cifar10':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        else:
            print("data type is incorrect")
            raise NotImplementedError
        
        if args.model == 'H':
            net = BetaVAE
        else:
            raise NotImplementedError('only support model H or B')
        
        ## define network and optimizer
        self.net = net(self.out_dim,self.z_dim, self.nc).to(self.device)

        self.optim = optim.Adam(self.net.parameters(), lr=self.lr,
                                    betas=(self.beta1, self.beta2))

        self.viz_name = args.viz_name

        self.ckpt_dir = os.path.join(args.ckpt_dir, args.viz_name)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)

        self.ckpt_name = args.ckpt_name
        ## load checkpoint
        if self.ckpt_name is not None and self.load_ckp:
            self.load_checkpoint(self.ckpt_name)

        self.save_output = args.save_output
        self.output_dir = os.path.join(args.output_dir, args.viz_name)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        ##----------------------------------------
        ## Step 1: finish code for data loader
        ##----------------------------------------
        self.data_loader = return_data(args)
        
    def train(self):
        self.net_mode(train=True)
        out = False
        
        pbar = tqdm(total=self.max_iter)
        pbar.update(self.global_iter)
        ## write log to log file
        with open('loss.txt','w') as loss_file:
            while not out:
                train_loss = 0
                for x, y in self.data_loader:
                    self.global_iter += 1
                    pbar.update(1)
                    x = Variable(x).to(self.device)
                    ##----------------------------------------
                    ## Step 2: feed image x into VAE network
                    ##----------------------------------------
                    x_recon, mu, logvar = self.net(x)
                    ## then compute the recon loss and KL loss
                    recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)
                    total_kld, dim_wise_kld = kl_divergence(mu, logvar)
                    
                    ##------------------------------------------------
                    ## Step 3: write code of loss/objective function
                    ##------------------------------------------------
                    vae_loss = recon_loss + 1*total_kld

                    if self.global_iter % 200 ==0:
                        print("vae_loss:{} recon_loss:{} KL_loss:{}".format(vae_loss.item(),recon_loss.item(),total_kld.item()))
                    
                    ##------------------------------------------
                    ## Step 4: write code of back propagation
                    ##------------------------------------------
                    self.optim.zero_grad()
                    vae_loss.backward()
                    train_loss += vae_loss.item()
                    self.optim.step() 

                    ## visualize the images
                    if (self.global_iter) % self.save_step ==0:
                        print("visualize the images")
                        self.viz_traverse()

                    if self.global_iter % self.save_step == 0:
                        self.save_checkpoint('last')
                        pbar.write('Saved checkpoint(iter:{})'.format(self.global_iter))
                    
                    if self.global_iter >= self.max_iter:
                        out = True
                        break
                print("training loss: " + str(train_loss/len(self.data_loader)))
                loss_file.write(str(train_loss/len(self.data_loader)))
                loss_file.write("\n")
        
        pbar.write("[Training Finished]")
        pbar.close()
    

    def viz_traverse(self):
        self.net_mode(train=False)
        decoder = self.net.decoder
        
        if self.save_output:
            output_dir = os.path.join(self.output_dir, str(self.global_iter))
            print("save test results: ", output_dir)
            os.makedirs(output_dir, exist_ok=True)

        for i in range(10):
            ##-------------------------------------------------------------
            ## Step 5: randomly/uniformly generate z as the input of decoder
            ##-------------------------------------------------------------
            z = torch.randn(60, self.z_dim).cuda()

            ## get the ouput of reconstructed image
            sample = torch.sigmoid(decoder(z))
            save_image(tensor=sample.cpu(),fp=os.path.join(output_dir, '{}_image.jpg'.format(i)),\
                    nrow=self.z_dim, pad_value=1)

        ## switch back to model train
        self.net_mode(train=True)

    
    def net_mode(self, train):
        if not isinstance(train, bool):
            raise('Only bool type is supported. True or False')

        if train:
            self.net.train()
        else:
            self.net.eval()

    def save_checkpoint(self, filename, silent=True):
        model_states = {'net':self.net.state_dict(),}
        optim_states = {'optim':self.optim.state_dict(),}
        states = {'iter':self.global_iter,
                  'model_states':model_states,
                  'optim_states':optim_states}

        file_path = os.path.join(self.ckpt_dir, filename)
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)
        if not silent:
            print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))
            

    def load_checkpoint(self, filename):
        file_path = os.path.join(self.ckpt_dir, filename)
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            self.global_iter = checkpoint['iter']
            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))
            
        
            
