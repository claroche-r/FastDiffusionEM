# Fast Diffusion EM

[Arxiv version  of the article](https://arxiv.org/pdf/2309.00287.pdf)

This repository implement the code for the article: **Fast Diffusion EM: a diffusion model for blind inverse problems with application to deconvolution.**

## Abstract
Using diffusion models to solve inverse problems is a growing field of research. Current methods assume the degradation to be known and provide impressive results in terms of restoration quality and diversity. In this work, we leverage the efficiency of those models to jointly estimate the restored image and unknown parameters of the degradation model. In particular, we designed an algorithm based on the well-known Expectation-Minimization (EM) estimation method and diffusion models. Our method alternates between approximating the expected log-likelihood of the inverse problem using samples drawn from a diffusion model and a maximization step to estimate unknown model parameters. For the maximization step, we also introduce a novel blur kernel regularization based on a Plug \& Play denoiser. Diffusion models are long to run, thus we provide a fast version of our algorithm. Extensive experiments on blind image deblurring demonstrate the effectiveness of our method when compared to other state-of-the-art approaches.

## Run model

**Setup:**

For the diffusion model, feel free to use your favorite pre-trained model. In our experiments we used the model pre-trained by the authors of [DPS](https://github.com/DPS2022/diffusion-posterior-sampling). 

For the blur kernel Plug & Play regularization, the weights can be found here: https://drive.google.com/drive/folders/1pueQC9FI0ozoSUiu4u1MlJR52zYSnvKr?usp=share_link

In the paper's experiments, we used the kernel_denoiser_33.pth model which is trained on 33x33 blur kernels. Using the 64x64 model might requires some tuning.

**Diffusion EM:**
``` 
python test_DiffEM.py --diffusion_config 'configs/diffusion_config_pigdm.yaml' --input_dir 'testset' --save_dir './results'
```
you can also replace "pigdm" diffusion by "dps". "pigdm" correspond to [$\Pi$ GDM](https://openreview.net/forum?id=9_gsMA8MRKQ) backbone while "dps" correspond to [DPS](https://openreview.net/forum?id=OnD9zGAGT0k) diffusion.

**Fast Diffusion EM:**
``` 
python test_FastDiffEM.py --diffusion_config 'configs/diffusion_config_fastem_pigdm.yaml' --input_dir 'testset' --save_dir './results'
```
you can also replace "pigdm" diffusion by "dps" in the code.

## Acknowledgement
The codes use [DPS](https://github.com/DPS2022/diffusion-posterior-sampling) as code base.

Our blur kernels are available for download [here](https://drive.google.com/file/d/1o1ruvDSbR9R12DzjA-2KIps7cqy4544v/view?usp=share_link).

## Citation
If you use our work, please cite us with the following:
```
@InProceedings{laroche2024fastem,
  title = {Fast Diffusion EM: a diffusion model for blind inverse problems with application to deconvolution},
  author = {Laroche, Charles and Almansa, Andr\'{e}s and Eva Coupet\'{e}},
  booktitle = {IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)}
  year = {2024}
}
```
