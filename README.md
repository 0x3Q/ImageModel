# Image Model

## Dataset

We use the [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) dataset, which contains 900 high-resolution images. Before training, we extract random **256x256** patches from both training and validation set, using: `python tooling/import.py <IMAGES DIRECTORY> <OUTPUT DIRECTORY>`.

## References:

- Agustsson E., Timofte R. (2017) [NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study](http://www.vision.ee.ethz.ch/~timofter/publications/Agustsson-CVPRW-2017.pdf)
- Ballé J., Laparra V., Simoncelli E. P. (2016) [Density modeling of images using a generalized normalization transformation](https://arxiv.org/abs/1511.06281).
- Ballé J., Laparra V., Simoncelli E. P. (2016) [End-to-end optimization of nonlinear transform codes for perceptual quality](https://arxiv.org/abs/1607.05006).
- Ballé J., Minnen D., Singh S., Hwang S. J., Johnston N. (2018) [Variational image compression with a scale hyperprior](https://arxiv.org/abs/1802.01436).
- Minnen D., Ballé J., Toderici G. (2018) [Joint autoregressive and hierarchical priors for learned image compression](https://arxiv.org/abs/1809.02736).
- Van den Oord A., Vinyals O., Kavukcuoglu K. (2017) [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)
