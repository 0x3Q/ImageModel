use clap::{Parser, Subcommand};
use reqwest::blocking::Client;
use sha2::{Digest, Sha256};

#[derive(Parser)]
struct Arguments {
    #[command(subcommand)]
    command: CommandType,
}

#[derive(Subcommand)]
enum CommandType {
    FetchImages { count: i64, directory: String },
}

fn main() -> Result<(), String> {
    let arguments = Arguments::parse();

    match arguments.command {
        CommandType::FetchImages { count, directory } => {
            let client = Client::new();
            for _ in 0..count {
                save_image_from_endpoint(&client, "https://picsum.photos/256", &directory)?;
            }
        }
    }

    Ok(())
}

fn save_image_from_endpoint(
    client: &Client,
    endpoint: &str,
    directory: &str,
) -> Result<(), String> {
    let response = client
        .get(endpoint)
        .send()
        .map_err(|error| format!("Error occured while sending request to {endpoint}: {error}"))?
        .error_for_status()
        .map_err(|error| format!("Request failed: {error}"))?;

    let content_type = response
        .headers()
        .get("content-type")
        .ok_or(format!("Unable to find content-type in headers!"))?
        .to_str()
        .map_err(|error| format!("Unable to convert content-type into string: {error}"))?
        .to_owned();

    let content_bytes = response.bytes().map_err(|error| format!("{error}"))?;

    match content_type.as_str() {
        "image/png" => save_image_without_duplicates(directory, "png", &content_bytes),
        "image/jpg" | "image/jpeg" => {
            save_image_without_duplicates(directory, "jpg", &content_bytes)
        }
        _ => Err(format!("Unsupported content-type: {content_type}!")),
    }
}

fn save_image_without_duplicates(
    directory: &str,
    extension: &str,
    bytes: &[u8],
) -> Result<(), String> {
    let file_hash = Sha256::digest(bytes);
    let file_name = format!("{file_hash:x}.{extension}");
    let file_path = std::path::Path::new(directory).join(&file_name);

    match file_path.exists() {
        true => {
            println!("Image exists! Skipping...");
            Ok(())
        }
        false => {
            println!("Image is being saved: {file_name}...");
            std::fs::write(file_path, bytes)
                .map_err(|error| format!("Error occured while writing to {file_name}: {error}"))
        }
    }
}
