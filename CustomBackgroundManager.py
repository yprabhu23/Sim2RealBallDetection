import os
import random
from PIL import Image
import numpy as np
import time

class RandomIndoorBackgroundGenerator:
    """
    Generate a random set of images from a dataset of indoor backgrounds"""

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.backgrounds = []


    def load_random_backgrounds(self, n):
        """
        Load a random set of background images from random folders within dataset_path.

        Returns:
            List[np.ndarray]: List of images as (H, W, C) NumPy arrays.
        """
        self.backgrounds = []

        subfolders = [
            os.path.join(self.dataset_path, d)
            for d in os.listdir(self.dataset_path)
            if os.path.isdir(os.path.join(self.dataset_path, d))
        ]

        for _ in range(n):
            # Choose a random subfolder
            folder = random.choice(subfolders)

            # List image files in the folder
            image_files = [
                f for f in os.listdir(folder)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
            ]

            if not image_files:
                continue  # skip empty folders

            # Choose a random image file
            img_path = os.path.join(folder, random.choice(image_files))

            # Load and store the image
            try:
                img = Image.open(img_path) 
                self.backgrounds.append(np.array(img))
            except Exception as e:
                print(f"Failed to load {img_path}: {e}")
                continue

        return self.backgrounds

    def visualize_backgrounds(self):
        """
        Visualize the loaded backgrounds using PIL.
        """
        for img in self.backgrounds:
            img_pil = Image.fromarray(img)
            img_pil.show()
            time.sleep(1) # Optional delay
            


if __name__ == "__main__":

    dataset_path = "./Images/"

    # Make sure it's an absolute path (optional but safer)
    abs_dataset_path = os.path.abspath(dataset_path)

    background_generator = RandomIndoorBackgroundGenerator(abs_dataset_path)

    backgrounds = background_generator.load_random_backgrounds(5)

    background_generator.visualize_backgrounds()



