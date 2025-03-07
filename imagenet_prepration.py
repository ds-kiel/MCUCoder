import os
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm 

def get_image_resolutions(image_folder):
    """
    Traverse through the image folder and get the resolution (area) of each image.
    
    Parameters:
    - image_folder (str): The directory where images are stored.
    
    Returns:
    - list of tuples: Each tuple contains (image_path, resolution).
    """
    resolutions = []
    for root, dirs, files in tqdm(os.walk(image_folder)):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                try:
                    with Image.open(image_path) as img:
                        width, height = img.size
                        resolutions.append((image_path, width * height))
                except Exception as e:
                    print(f"Error opening image {image_path}: {e}")
    return resolutions

def select_largest_images(resolutions, num_images):
    """
    Select the images with the largest resolutions.
    
    Parameters:
    - resolutions (list of tuples): List containing (image_path, resolution).
    - num_images (int): Number of top images to select based on resolution.
    
    Returns:
    - list of str: Paths to the selected images.
    """
    sorted_images = sorted(resolutions, key=lambda x: x[1], reverse=True)
    return [image[0] for image in sorted_images[:num_images]]

def add_random_noise(image):
    """
    Add uniform random noise to the image.
    
    Parameters:
    - image (PIL.Image): The image to which noise will be added.
    - noise_level (float): The maximum noise level. Noise will be uniformly distributed in the range [-noise_level, noise_level].
    
    Returns:
    - PIL.Image: The image with added noise.
    """
    img_array = np.array(image)
    noise = np.random.uniform(0, 1, img_array.shape).astype(np.uint8)
    noisy_img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img_array)

def resize_image(image, width, height):
    """
    Resize the image if its dimensions are larger than 512x512.
    
    Parameters:
    - image (PIL.Image): The image to be resized.
    - width (int): The width of the image.
    - height (int): The height of the image.
    
    Returns:
    - PIL.Image: The resized image.
    """
    if min(width, height) > 512:
        new_width = int(width // 2)
        new_height = int(height // 2)
        img_array = np.array(image)
        resized_img_array = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(resized_img_array)
    return image

def save_selected_images(image_paths, output_folder):
    """
    Save the selected images as PNG files in the specified output folder after adding uniform random noise.
    
    Parameters:
    - image_paths (list of str): Paths to the images to be saved.
    - output_folder (str): Directory where the images will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for image_path in image_paths:
        try:
            base_name = os.path.basename(image_path)
            name, _ = os.path.splitext(base_name)
            output_path = os.path.join(output_folder, f"{name}.png")  # Ensure the output file is PNG
            with Image.open(image_path) as img:
                width, height = img.size
                img = resize_image(img, width, height)
                img = add_random_noise(img)
                img.convert("RGB").save(output_path, format="PNG")  # Convert to RGB if necessary and save as PNG
        except Exception as e:
            print(f"Error saving image {image_path}: {e}")

def main():
    image_folder = '/data22/datasets/imagenet/train/'  # Path to the folder containing ImageNet images
    output_folder = '/data22/aho/high_res_imagenet/'  # Path where the selected images will be saved
    num_images = 300_000
    
    print("Getting image resolutions...")
    resolutions = get_image_resolutions(image_folder)
    
    print(f"Selecting {num_images} largest images...")
    largest_images = select_largest_images(resolutions, num_images)
    
    print(f"Saving selected images to {output_folder}...")
    save_selected_images(largest_images, output_folder)
    
    print("Done!")
if __name__ == "__main__":
    main()
