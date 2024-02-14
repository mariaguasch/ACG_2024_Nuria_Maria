from google_images_download import google_images_download
import os

def download_similar_images(seed_image_path, save_directory, num_images=10):
    # Create save directory if it doesn't exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Perform reverse image search
    response = google_images_download.googleimagesdownload()
    arguments = {"keywords": "visually similar images", "limit": num_images, "url": seed_image_path, "output_directory": save_directory}
    paths = response.download(arguments)

    # Print paths of downloaded images
    print(paths)

# Example usage
seed_image_path = "/Users/nuriacodina/Desktop/UPF/QUART/2N_TRIM/FGA/ACG_2024_Nuria_Maria/lab4/TRAINING/image_A0011.jpg"
save_directory = "/Users/nuriacodina/Desktop/UPF/QUART/2N_TRIM/FGA/ACG_2024_Nuria_Maria/lab4/new_images"
num_images = 5
download_similar_images(seed_image_path, save_directory, num_images)