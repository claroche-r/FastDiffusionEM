from functools import partial
import os
import argparse
import yaml
import pickle

import numpy as np
import torch

from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler

from util.logger import get_logger
import util.utils_image as util


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    # Configurations
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_model_config', type=str, default='configs/model_config.yaml')
    parser.add_argument('--diffusion_config', type=str, default='configs/diffusion_config_fastem_pigdm.yaml')

    # Training
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--input_dir', type=str, default='./testset')
    parser.add_argument('--save_dir', type=str, default='./results')

    args = parser.parse_args()
   
    # logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    img_model_config = load_yaml(args.img_model_config)
    diffusion_config = load_yaml(args.diffusion_config)
   
    # Load model
    img_model = create_model(**img_model_config)
    img_model = img_model.to(device)
    img_model.eval()

    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    sample_fn = partial(sampler.p_sample_loop, model=img_model)
   
    # Working directory
    out_path = args.save_dir
    logger.info(f"work directory is created as {out_path}")
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # set seed for reproduce
    np.random.seed(123)


    dir = args.input_dir
    paths = os.listdir(dir)

    for path in paths:
        with open(os.path.join(dir, path), 'rb') as f:
            sample = pickle.load(f)

        y_n = sample['L'].to('cuda')
        H = sample['H']
        ker = sample['kernel']
        sigma = sample['sigma']
        
        fname = os.path.split(path)[-1][:-7]

        # Set initial sample 
        x_start = torch.randn((1, y_n.shape[1], y_n.shape[2], y_n.shape[3]), device=device).requires_grad_()
        
        # sample 
        sample = sample_fn(x_start=x_start, measurement=y_n, record=False, save_root=out_path, sigma=sigma)

        est = sample[0].detach().cpu()
        ker_est = sample[1].detach().cpu()
        
        # GT
        util.imwrite(util.tensor2uint(H), os.path.join(os.path.join(out_path, 'label') ,  fname + '_H.png'))
        util.imwrite(util.tensor2uint(ker/ker.max()), os.path.join(os.path.join(out_path, 'label') ,  fname + '_ker.png'))
        # Input
        util.imwrite(util.tensor2uint(y_n), os.path.join(os.path.join(out_path, 'input') ,  fname + '.png'))
        # Results
        util.imwrite(util.tensor2uint(est), os.path.join(os.path.join(out_path, 'recon') ,  fname + '_E.png'))
        util.imwrite(util.tensor2uint(ker_est/ker_est.max()), os.path.join(os.path.join(out_path, 'recon') ,  fname + '_ker.png'))

if __name__ == '__main__':
    main()
