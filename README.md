# Image Model

## Tooling

### Commands:

- **fetch-images:** Downloads a specified number of images from the internet and
  saves them to an output directory. Duplicates are automatically skipped based
  on a SHA-256 hash.
  - **Usage:**
    ```bash
    cargo run --manifest-path tooling/Cargo.toml -- fetch-images <COUNT> <OUTPUT DIRECTORY>
    ```
  - **Example:**
    ```bash
    # This will fetch 10 images and save them in the 'images' directory
    cargo run --manifest-path tooling/Cargo.toml -- fetch-images 10 ./images
    ```

## References:

- Ballé J., Minnen D., Singh S., Hwang S. J., Johnston N. (2018) [Variational image compression with a scale hyperprior](https://arxiv.org/abs/1802.01436).
- Minnen D., Ballé J., Toderici G. (2018) [Joint autoregressive and hierarchical priors for learned image compression](https://arxiv.org/abs/1809.02736).
- Ballé J., Laparra V., Simoncelli E. P. (2016) [Density modeling of images using a generalized normalization transformation](https://arxiv.org/abs/1511.06281).
- Ballé J., Laparra V., Simoncelli E. P. (2016) [End-to-end optimization of nonlinear transform codes for perceptual quality](https://arxiv.org/abs/1607.05006).
- Van den Oord A., Vinyals O., Kavukcuoglu K. (2017) [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)
