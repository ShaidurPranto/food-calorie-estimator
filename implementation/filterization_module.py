# class of filterization

import os
import shutil
import numpy as np
import torch
import timm
import matplotlib.pyplot as plt
from PIL import Image
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

class FoodFilter:
    """
    A filterization module to classify whether an image contains food or not.
    """
    def __init__(self):
        """
        Initialize the FoodFilter.
        """
        self.model_id = "mrdbourke/food-not-food-classifier-csatv2-v1"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load Model Architecture
        self.model = timm.create_model("csatv2.r512_in1k", pretrained=False, num_classes=2)
        
        # Download and Load Weights
        weights_path = hf_hub_download(repo_id=self.model_id, filename="model.safetensors")
        state_dict = load_file(weights_path)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # Prepare Preprocessing
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)

    def is_food(self, image_path: str, show_plot: bool = True) -> bool:
        """
        Predicts whether an image is food or not.
        
        Args:
            image_path (str): path to the image
            show_plot (bool): whether to display the image and prediction using matplotlib
            
        Returns:
            bool: True if probability of being food is > 0.5, False otherwise.
        """
        # img = Image.open(image_path).convert("RGB")
        # img = Image.open(image_path)

        img_raw = Image.open(image_path)
        # If the image has an alpha channel (transparency)
        if img_raw.mode in ("RGBA", "P"):
            img_raw = img_raw.convert("RGBA")
            # Create a white background (or black, depending on your model's training)
            background = Image.new("RGB", img_raw.size, (255, 255, 255)) 
            background.paste(img_raw, mask=img_raw.split()[3]) # 3 is the alpha channel
            img = background
        else:
            img = img_raw.convert("RGB")


        
        img_tensor = self.transforms(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
        # Index 0 is food, Index 1 is not food
        food_prob = float(probabilities[0])
        is_food_pred = food_prob > 0.5
        
        if show_plot:
            plt.figure(figsize=(6, 6))
            plt.imshow(img)
            plt.title(f"Prediction: {'Food' if is_food_pred else 'Not Food'} (Prob: {food_prob:.4f})")
            plt.axis('off')
            plt.show()

        return is_food_pred

    def get_food_array(self, folder_path: str) -> np.ndarray:
        """
        Evaluates all images in a folder and returns a NumPy array of 0s and 1s,
        where 1 indicates the image contains food, and 0 otherwise. The array length
        matches the number of images found in the folder.
        
        Args:
            folder_path (str): path to the folder containing images
            
        Returns:
            np.ndarray: Array of 0s and 1s denoting non-food and food respectively.
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")
            
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
        files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)])
        
        food_array = np.zeros(len(files), dtype=int)
        for index, file_name in enumerate(files):
            image_path = os.path.join(folder_path, file_name)
            try:
                if self.is_food(image_path):
                    food_array[index] = 1
            except Exception as e:
                print(f"Skipping {file_name} due to an error: {e}")
                
        return food_array

    def get_food_arrays_for_subfolders(self, root_folder_path: str) -> list:
        """
        Evaluates images within all subfolders of the given root folder.
        Prints the folder structure and evaluates the files.
        
        Args:
            root_folder_path (str): path to the root folder which contains multiple subfolders.
            
        Returns:
            list: A list of NumPy arrays (one for each subfolder) containing 0s and 1s,
                  where 1 indicates the image contains food, and 0 otherwise.
        """
        if not os.path.exists(root_folder_path):
            raise FileNotFoundError(f"Root folder not found: {root_folder_path}")
            
        subfolders = [f for f in os.listdir(root_folder_path) 
                      if os.path.isdir(os.path.join(root_folder_path, f))]
        subfolders = sorted(subfolders)
        
        print(f"\nFolder structure for: {root_folder_path}")
        all_food_arrays = []
        for subfolder in subfolders:
            subfolder_path = os.path.join(root_folder_path, subfolder)
            
            # Print the directory structure
            print(f"├── {subfolder}/")
            try:
                files = sorted(os.listdir(subfolder_path))
                for file_name in files:
                    if os.path.isfile(os.path.join(subfolder_path, file_name)):
                        print(f"│   ├── {file_name}")
                        
                food_array = self.get_food_array(subfolder_path)
                all_food_arrays.append(food_array)
            except Exception as e:
                print(f"Error processing subfolder {subfolder}: {e}")
                
        print("\n")
        return all_food_arrays

    def filter_folders_by_food_arrays(self, input_folder_path: str, food_arrays: list, output_folder_path: str) -> None:
        """
        Copies files from subfolders of the input folder to the output folder based on a 1/0 mapping.
        Only files corresponding to a '1' in the food_arrays are copied.
        
        Args:
            input_folder_path (str): path to the folder containing subfolders of files.
            food_arrays (list): list of 1D arrays/lists (0s and 1s) representing food or not, for each subfolder.
            output_folder_path (str): path where the filtered subfolders and their files will be saved.
        """
        if not os.path.exists(input_folder_path):
            raise FileNotFoundError(f"Input folder not found: {input_folder_path}")
            
        os.makedirs(output_folder_path, exist_ok=True)
        
        # Get and sort subfolders alphabetically to match the generation order
        subfolders = [f for f in os.listdir(input_folder_path) 
                      if os.path.isdir(os.path.join(input_folder_path, f))]
        subfolders = sorted(subfolders)
        
        if len(subfolders) != len(food_arrays):
            print(f"Warning: Extracted {len(subfolders)} subfolders, but found {len(food_arrays)} food arrays.")
            
        for i, subfolder in enumerate(subfolders):
            if i >= len(food_arrays):
                break
                
            input_subfolder_path = os.path.join(input_folder_path, subfolder)
            output_subfolder_path = os.path.join(output_folder_path, subfolder)
            
            # Create subfolder in output directory
            os.makedirs(output_subfolder_path, exist_ok=True)
            
            # Get files within the subfolder (all extensions as requested)
            files = [f for f in os.listdir(input_subfolder_path) 
                     if os.path.isfile(os.path.join(input_subfolder_path, f))]
            files = sorted(files)
            
            food_array = food_arrays[i]
            
            for j, file_name in enumerate(files):
                # Ensure we don't go out of bounds if there's a mismatch
                if j >= len(food_array):
                    break
                    
                if food_array[j] == 1:
                    src_file = os.path.join(input_subfolder_path, file_name)
                    dst_file = os.path.join(output_subfolder_path, file_name)
                    shutil.copy2(src_file, dst_file)

