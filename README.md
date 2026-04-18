# Image Model

## Datasets

We use the LSDIR dataset, which contains 84,991 high-quality training images. During training, we extract random 256x256 patches and apply random horizontal flips for augmentation. We evaluate our model performance using the LSDIR validation dataset by extracting multiple patches from each image and calculating averages of the metrics.

## References:

- Agustsson E., Timofte R. (2017) [NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study](http://www.vision.ee.ethz.ch/~timofter/publications/Agustsson-CVPRW-2017.pdf)
- Ballé J., Laparra V., Simoncelli E. P. (2016) [Density modeling of images using a generalized normalization transformation](https://arxiv.org/abs/1511.06281).
- Ballé J., Laparra V., Simoncelli E. P. (2016) [End-to-end optimization of nonlinear transform codes for perceptual quality](https://arxiv.org/abs/1607.05006).
- Ballé J., Minnen D., Singh S., Hwang S. J., Johnston N. (2018) [Variational image compression with a scale hyperprior](https://arxiv.org/abs/1802.01436).
- Franzen R. (1999) [Kodak Lossless True Color Image Suite](https://r0k.us/graphics/kodak/)
- Li Y., Zhang K., Liang J., Cao J., Liu C., Gong R., Zhang Y., Tang H., Liu Y., Demandolx D., Ranjan R., Timofte R., Van Gool L. (2023) [LSDIR: A large scale dataset for image restoration](https://ieeexplore.ieee.org/document/10208419)
- Lim B., Son S., Kim H., Nah S., Lee K. M. (2017) [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://ieeexplore.ieee.org/document/8014885)
- Liu Z., Lin Y., Cao Y., Hu H., Wei Y., Zhang Z., Lin S., Guo B. (2021) [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)
- Minnen D., Ballé J., Toderici G. (2018) [Joint autoregressive and hierarchical priors for learned image compression](https://arxiv.org/abs/1809.02736).
- Van den Oord A., Vinyals O., Kavukcuoglu K. (2017) [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)
