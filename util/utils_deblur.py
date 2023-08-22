import torch
import util.utils_agem as agem


def hadamard(x, kmap):
    # Compute hadamard product (pixel-wise)
    # x: input of shape (C,H,W)
    # kmap: input of shape (H,W)

    C,H,W = x.shape
    kmap = kmap.view(1, H, W)
    kmap = kmap.repeat(C, 1, 1)
    return (x * kmap)

def o_leary(x, kmap, basis):
    # Apply O'Leary convolution model blurry = sum(U_i H_i x)
    # x: input of shape (C,H,W)
    # kmap: input of shape (P,H,W)
    # basis: input of shape (P,H_k,W_k)

    assert len(kmap) == len(basis), str(len(kmap)) + ',' +  str(len(basis))
    c = 0
    for i in range(len(kmap)):
        k = basis[i][None]
        c += hadamard(agem.fft_blur(x[None], k[None])[0], kmap[i])
    return c

def o_leary_batch(x, kmap, basis):
    # Apply O'Leary convolution model blurry = sum(U_i H_i x)
    # x: input of shape (B,C,H,W)
    # kmap: input of shape (B,P,H,W)
    # basis: input of shape (B,P,H_k,W_k)

    assert len(x) == len(kmap) and len(kmap) == len(basis), print("Batch size must be the same for all inputs")
    
    return torch.cat([o_leary(x[i], kmap[i], basis[i])[None] for i in range(len(x))])

def transpose_o_leary(x, kmap, basis):
    # Apply the transpose of O'Leary convolution model blurry = sum(H_i^T U_i x)
    # x: input of shape (C,H,W)
    # kmap: input of shape (P,H,W)
    # basis: input of shape (P,H_k,W_k)
    
    assert len(kmap) == len(basis), str(len(kmap)) + ',' +  str(len(basis))
    c = 0
    for i in range(len(kmap)):
        k = torch.flip(basis[i], dims =(0,1))[None]
        c += agem.fft_blur(hadamard(x, kmap[i])[None], k[None])[0]
    return c

def transpose_o_leary_batch(x, kmap, basis):
    # Apply the transpose of O'Leary convolution model blurry = sum(H_i^T U_i x)
    # x: input of shape (B,C,H,W)
    # kmap: input of shape (B,P,H,W)
    # basis: input of shape (B,P,H_k,W_k)

    assert len(x) == len(kmap) and len(kmap) == len(basis), print("Batch size must be the same for all inputs")
    
    return torch.cat([transpose_o_leary(x[i], kmap[i], basis[i])[None] for i in range(len(x))])
