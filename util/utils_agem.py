import torch
import numpy as np
import torch.fft as fft
from math import sqrt
from scipy.ndimage import measurements, interpolation
import util.utils_deblur as deblur

def clean_output(x):
    return (x.clamp(-1, 1) + 1) / 2

def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())

def p2o(psf, shape):
    '''
    Convert point-spread function to optical transfer function.
    otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.

    Args:
        psf: NxCxhxw
        shape: [H, W]

    Returns:
        otf: NxCxHxWx2
    '''
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[...,:psf.shape[2],:psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis+2)
    return otf

def o2p(otf, shape):
    shape_otf = otf.shape[-2:]
    psf = otf
    for axis, axis_size in enumerate(shape):
        psf = torch.roll(psf, int(axis_size / 2), dims=axis+2)
    return psf[:,:,:shape[0], :shape[1]]

def fft_blur(img, k):
    k = p2o(k, img.shape[-2:])
    img_fft = fft.fft2(img)
    ker_fft = fft.fft2(k)

    res_fft = ker_fft.mul(img_fft)
    res = fft.ifft2(res_fft)
    return res.real


def data_solution_deblur(z, x, y, alpha):
    Fx = fft.fft2(x)
    FxC = Fx.conj()
    Fx_sumC = FxC.sum(axis=0)[None]
    F2x = Fx.mul(FxC)
    F2x_sum = F2x.sum(axis=0)[None]
    Fx_sumFy = Fx_sumC.mul(fft.fft2(y))
    res_fft = (Fx_sumFy + fft.fft2(alpha * z)) / (F2x_sum + alpha)
    return fft.ifft2(res_fft).real

def normalize_kernel(k):
    _, _, h, w = k.shape
    k = torch.maximum(torch.zeros_like(k), k)
    k /= k.sum(axis=(-2,-1))[:,:, None,None].repeat(1,1,h,w)

    return k

def regularization_step(k, lamb, beta, denoiser=None, how='l1'):
    _, _, n,p = k.shape
    if how == 'l1':
        res = k - (lamb / beta)
        res = torch.maximum(torch.zeros_like(res), res)
        
    elif how == 'l2':
        res = (beta * k) / (2 * lamb + beta)
        
    elif how == 'pnp':
        if not lamb ==0:
            sigma_d = torch.ones_like(k)[:,:1,:,:] * (sqrt(lamb/beta))
            res = denoiser(torch.cat((k, sigma_d), axis=1))
        else:
            res = k
    return normalize_kernel(res)

def HQS_ker(y, x, sigma, init_ker, denoiser=None, reg='pnp',
            beta=1e5, lamb=1, n_iter=100, ksize=33):
    
    alpha = sigma**2 * beta
    k0 = init_ker
    z0 = k0
    
    for i in range(n_iter):
        k0 = data_solution_deblur(p2o(z0,y.shape[-2:]), x, y, alpha).mean(axis=1)[:,None,:,:]
        k0 = o2p(k0, (ksize, ksize))
        k0 = normalize_kernel(k0)
        z0 = regularization_step(k0, lamb, beta, denoiser=denoiser, how=reg)
        z0 = normalize_kernel(z0)
            
    return z0

def ADMM_ker(y, x, sigma, init_ker, denoiser=None, reg='pnp',
            beta=1e5, lamb=1, n_iter=100, ksize=(33,33)):
    
    alpha = sigma**2 * beta
    k0 = init_ker
    z0 = k0
    u0 = k0 - z0
    
    
    for i in range(n_iter):
        k0 = data_solution_deblur(p2o((z0 - u0),y.shape[-2:]), x, y, alpha).mean(axis=1)[:,None,:,:]
        k0 = o2p(k0, ksize)
        #k0 = normalize_kernel(k0)
        z0 = regularization_step(k0 + u0, lamb, beta, denoiser=denoiser, how=reg)
        z0 = normalize_kernel(z0)
        u0 = u0 + (k0 - z0)
            
    return z0


def pad_kernel(k, ksize):
    pad_values = (ksize[0] - k.shape[-2]) // 2, (ksize[1] - k.shape[-1]) // 2
    k = torch.nn.functional.pad(k, (pad_values[1],pad_values[1], pad_values[0],pad_values[0]),
                                     mode='constant', value=0)
    return k

def deblurring_guidance(y, x, k, sigma=0, r=1):
    if sigma == 0:
        sigma += 1e-8
        
    k = p2o(k, x.shape[-2:])
    Fk = fft.fft2(k)
    FkC = Fk.conj()
    Fk2 = Fk.mul(FkC)
    Fy = fft.fft2(y)
    Fx = fft.fft2(x)

    num = FkC.mul(Fy - Fk.mul(Fx))

    if r==0:
        out = (num / sigma**2)
    else:
        den = r ** 2 * Fk2 + sigma ** 2
        out = num / den

    return fft.ifft2(out).real


def pinv_deblurring(x, k, r=1, sigma=1e-8):
    k = p2o(k, x.shape[-2:])
    Fk = fft.fft2(k)
    FkC = Fk.conj()
    Fk2 = Fk.mul(FkC)
    Fx = fft.fft2(x)
    num = FkC
    den = r ** 2 * Fk2 + sigma ** 2
    res = (num / den).mul(Fx)
    return fft.ifft2(res).real


def deblurring_guidance_bis(y, x, k, sigma=0, r=1):
    if sigma == 0:
        sigma += 1e-8
    return pinv_deblurring(y, k, r=r, sigma=sigma) - pinv_deblurring(fft_blur(x, k), k, r=r, sigma=sigma)

def pinv_oleary(x, kmap, basis, r=1, sigma=1e-8, how='max'):
    if how == 'max':
        k_max = get_max_conv_batch(kmap, basis)
        k_max = p2o(k_max, x.shape[-2:])
        Fk = fft.fft2(k_max)
        FkC = Fk.conj()
        Fk2 = Fk.mul(FkC)
        den = (r ** 2) * Fk2 + (sigma ** 2)

    elif how == 'mean':
        k_max = get_mean_conv_batch(kmap, basis)
        k_max = p2o(k_max, x.shape[-2:])
        Fk = fft.fft2(k_max)
        FkC = Fk.conj()
        Fk2 = Fk.mul(FkC)
        den = (r ** 2) * Fk2 + (sigma ** 2)

    elif how == 'min':
        k_max = get_min_conv_batch(kmap, basis)
        k_max = p2o(k_max, x.shape[-2:])
        Fk = fft.fft2(k_max)
        FkC = Fk.conj()
        Fk2 = Fk.mul(FkC)
        den = (r ** 2) * Fk2 + (sigma ** 2)
        
    num = fft.fft2(x)
    res = (num / den)
    res = fft.ifft2(res).real
    return deblur.transpose_o_leary_batch(res, kmap, basis)

def oleary_guidance(y, x, sigma, kmap, basis, r, how='max'):
    sigma += 1e-8
    return pinv_oleary(y, kmap, basis, sigma=sigma, r=r, how=how) - pinv_oleary(deblur.o_leary_batch(x, kmap, basis), kmap, basis, sigma=sigma, r=r, how=how)

def get_min_conv(kmap, basis):
    Fk_list = []
    for m, k in zip(kmap, basis):
        if m.sum() >0:
            Fk_list.append(k[None])

    return torch.min(torch.cat(Fk_list), dim=0)[0][None]

def get_min_conv_batch(kmap, basis):
    return torch.cat([get_min_conv(k, b)[None] for k,b in zip(kmap, basis)])

def get_max_conv(kmap, basis):
    Fk_list = []
    for m, k in zip(kmap, basis):
        if m.sum() > 0:
            Fk_list.append(k[None])
            
    return torch.max(torch.cat(Fk_list), dim=0)[0][None]

def get_max_conv_batch(kmap, basis):
    return torch.cat([get_max_conv(k, b)[None] for k,b in zip(kmap, basis)])

def get_mean_conv(kmap, basis):
    Fk_list = []
    for m, k in zip(kmap, basis):
        if m.sum() > 0:
            Fk_list.append(k[None])

    return torch.mean(torch.cat(Fk_list).real, dim=0)[None]

def get_mean_conv_batch(kmap, basis):
    return torch.cat([get_mean_conv(k, b)[None] for k,b in zip(kmap, basis)])

def kernel_shift(kernel, sf):
    # There are two reasons for shifting the kernel :
    # 1. Center of mass is not in the center of the kernel which creates ambiguity. There is no possible way to know
    #    the degradation process included shifting so we always assume center of mass is center of the kernel.
    # 2. We further shift kernel center so that top left result pixel corresponds to the middle of the sfXsf first
    #    pixels. Default is for odd size to be in the middle of the first pixel and for even sized kernel to be at the
    #    top left corner of the first pixel. that is why different shift size needed between odd and even size.
    # Given that these two conditions are fulfilled, we are happy and aligned, the way to test it is as follows:
    # The input image, when interpolated (regular bicubic) is exactly aligned with ground truth.

    # First calculate the current center of mass for the kernel
    current_center_of_mass = measurements.center_of_mass(kernel)

    # The second term ("+ 0.5 * ....") is for applying condition 2 from the comments above
    wanted_center_of_mass = np.array(kernel.shape) // 2 + 0.5 * (np.array(sf) - (np.array(kernel.shape) % 2))
    # Define the shift vector for the kernel shifting (x,y)
    shift_vec = wanted_center_of_mass - current_center_of_mass
    # Before applying the shift, we first pad the kernel so that nothing is lost due to the shift
    # (biggest shift among dims + 1 for safety)
    #pad_val = np.int(np.ceil(np.max(np.abs(shift_vec)))) + 1
    #kernel = np.pad(kernel, pad_val, 'constant')

    # Finally shift the kernel and return
    kernel = interpolation.shift(kernel, shift_vec)

    #kernel = kernel[...,pad_val:-pad_val, pad_val:-pad_val]
    return kernel

def crop_kernel(k, k_size):
    current_k_size = k.shape
    #print(current_k_size, k_size)
    if current_k_size == k_size:
        return k
    else:
        border_h = (current_k_size[0]-k_size[1]) // 2
        border_w = (current_k_size[1]-k_size[1]) // 2
        return k[border_h:-border_h, border_w:-border_w]

def load_network(net, path, device='cuda'):
    net.load_state_dict(torch.load(path))
    return net.to(device)
