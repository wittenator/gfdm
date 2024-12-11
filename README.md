<h1 align="center">Generative Fractional Diffusion Models <br> (NeurIPS 2024) </h1> 

This repository contains the official implementation of the paper [Generative Fractional Diffusion Models](https://arxiv.org/abs/2310.17638) (GFDM), introducing a continuous-time diffusion model driven by Markov approximate fractional Brownian motion, replacing the standard Brownian motion used in traditional diffusion models.

## Introduction
Fractional Brownian motion interpolates between the paths of Brownian-driven SDEs and those of the underlying integration in probability flow ODEs, while also offering even rougher paths:


![cover](visuals/thumbnail.png)

Our experiments demonstrate that, compared to purely Brownian dynamics, the super-diffusive (smooth) regime of Markov approximation of fractional Brownian motion achieves higher image quality with fewer score model evaluations, improved pixel-wise diversity and better distribution coverage.

## Dependencies

To run this code, install the latest project conda environment stored in `gfdm.yml` via 

```python
conda env create -f gfdm.yml
```

## Train on Custom Datasets 

You can use our repository to train GFDM on `mnist`, `fashionmnist` and `cifar10`. To train on your custom dataset add in `train.get_dataset` your dataset named `yourdataset` to the constructor:

```python
    constructor = {
        "mnist": vision_datasets.MNIST,
        "fashionmnist": vision_datasets.FASHIONMNIST,
        "cifar10": vision_datasets.CIFAR10,
        "yourdataset": CustomDataset
    }
```

Additionally, you need to add `yourdataset` to the available choices for `--data_name` in `args.py`. To use our code out-of-the-box, your `CustomDataset` must inherit from the `vision_datasets.TVData` class. To train with a Hurst index $H$ and $K$ augmenting processes on your dataset of size `(c,size,size)` with `num_classes` classes, use the following command:

```python
python train.py --data_name yourdataset --channels c --image_size size --num_classes num_classes --hurst H --num_aug K --dynamics fvp --train_steps 1000000
```

Depending on the size of your images, consider adjusting the following default arguments of `unet.UNetModel`: 
`--model_channels 128`, `--num_res_blocks 4`, `--attn_resolutions 8`,  `--channel_mult 1,2,2,2`.


## Choice of Hyperparameters in Diffusion Dynamics

For conditional image generation, we observe the best performance on MNIST and CIFAR-10 using Fractional Variance Preserving (FVP) dynamics with $H=0.9$, $K=3$ for MNIST, and $K=2$ for CIFAR-10:

<table>
<tr><th>CIFAR10 </th><th> MNIST</th></tr>
<tr><td>
    
| SDE   | FID $\downarrow$ | **$VS_{p}$ $\uparrow$**  |
|-------------------------|-------------|-------------|
| VE (retrained)          | $5.20$      | $3.42$      |
| VP (retrained)          | $4.85$      | $3.28$      |
| $\text{FVP}(H=0.9,K=1)$ | $4.79$      | $3.53$      |
| $\text{FVP}(H=0.7,K=2)$ | $4.17$      | $3.35$      |
| $\text{FVP}(H=0.9,K=2)$ | $3.77$      | $3.60$      |

</td><td>

| SDE     | FID $\downarrow$ | **$VS_{p}$ $\uparrow$** |
|-------------------------|-------------|--------------|
| VE (retrained)          | $10.82$     | $24.20$      |
| VP (retrained)          | $1.44$      | $23.64$      |
| $\text{FVP}(H=0.9,K=3)$ | $0.72$      | $24.18$      |
| $\text{FVP}(H=0.7,K=3)$ | $0.86$      | $24.39$      |
| $\text{FVP}(H=0.9,K=4)$ | $1.22$      | $24.76$      |

 </td></tr> </table>

 ## Reproduce Training

 ### On MNIST

 To train on MNIST we used the following parameters:
 
```python
python train.py --data_name mnist --channels 1 --image_size 28 --hurst 0.9 --num_aug 3 --dynamics fvp --model_channels 64 --num_res_blocks 3 --attn_resolutions 4,2 --channel_mult 1,2,4 --use_ema False --log_model_every_n 50000 --lr 1e-4 --batch_size 1024 --train_steps 50000 
```

 ### On CIFAR10

  To train on CIFAR10 we used the following parameters:

  ```python
  python train.py --data_name cifar10 --channels 3 --image_size 32 --hurst 0.9 --num_aug 2 --dynamics fvp --model_channels 128 --num_res_blocks 4 --attn_resolutions 8 --channel_mult 1,2,2,2 --use_ema True --log_model_every_n 100000 --lr 2e-4 --batch_size 128 --train_steps 1000000 
```
## Generate Data from Checkpoints
 
To generate $M$ samples over $N$ steps using either `--mode=sde` or `--mode ode` with the checkpoints at `path\model-{version}.pth` and `path\ema_model-{version}.pth` run: 

```python
python generate.py --data_name data_name --mode sde --steps N --n_samples M --batch_size batch_size --pth path
```

## Logging 

Our code uses Weights & Biases for looging. For online logging specify your personal key via `--wb_key your_wandb_key`. 

## Mulit-GPU training 
The code runs on as many GPUs as available. Consider to adjust `--batch_size` and `--accumulate_grad_batches` when swithing from one GPU to multiple-GPUs for an equivalent set-up.

## Bibtex Citation

We kindly ask that you cite our paper when using this code:
    
    @inproceedings{
    nobis2024generative,
    title={Generative Fractional Diffusion Models},
    author={Gabriel Nobis and Maximilian Springenberg and Marco Aversa and Michael Detzel and Rembert Daems and Roderick Murray-Smith and Shinichi Nakajima and Sebastian Lapuschkin and Stefano Ermon and Tolga Birdal and Manfred Opper and Christoph Knochenhauer and Luis Oala and Wojciech Samek},
    booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
    year={2024},
    url={https://openreview.net/forum?id=B9qg3wo75g}
    }
