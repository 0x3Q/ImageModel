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
