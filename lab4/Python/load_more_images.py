from google_images_download import google_images_download
import os
import requests
'''
def internet_connection():
    try:
        response = requests.get("https://dns.tutorialspoint.com", timeout=5)
        return True
    except requests.ConnectionError:
        return False    
if internet_connection():
    print("The Internet is connected.")
else:
    print("The Internet is not connected.")

internet_connection

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
seed_image_path = "TRAINING/image_A0006.jpg"
save_directory = "new_images"
num_images = 5
download_similar_images(seed_image_path, save_directory, num_images)'''

response = google_images_download.googleimagesdownload()   #class instantiation
o_direct = 'new_training_images'
arguments = {"limit":5, "output_directory":o_direct,'print_urls': True ,'similar_images':'TRAINING/image_A0006.jpg'}   #creating list of arguments
paths = response.download(arguments) 