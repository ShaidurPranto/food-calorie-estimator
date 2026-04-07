import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import timm
from torchvision import transforms

class FoodFilterV2:
    """
    A filterization module to classify whether an image contains food or not.
    This version directly utilizes the locally trained ViT model.
    """
    def __init__(self, confidence_threshold=0.3):
        """
        Initialize the FoodFilterV2.
        
        Args:
            confidence_threshold (float): Minimum confidence required from the 
                                          local model to classify as "Food".
        """
        self.model_path = "saved_models/model_1_vit_segment_aware.pth"
        self.num_classes = 19
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.confidence_threshold = confidence_threshold

        # Define image transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Load the model
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self):
        """
        Load the local model from checkpoint.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Create model architecture
        model = timm.create_model(
            "vit_small_patch16_224",
            pretrained=False,
            num_classes=self.num_classes
        )

        # Load state dict
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        return model

    def is_food(self, image_path: str, show_plot: bool = False) -> bool:
        """
        Predicts whether an image is food or not based on local model confidence.
        
        Args:
            image_path (str): path to the image
            show_plot (bool): whether to display the image and prediction using matplotlib
            
        Returns:
            bool: True if probability of being food is >= confidence_threshold, False otherwise.
        """
        img_raw = Image.open(image_path)
        
        # Handle alpha channel (transparency) gracefully
        if img_raw.mode in ("RGBA", "P"):
            img_raw = img_raw.convert("RGBA")
            # Create a white background to paste transparent pixels over
            background = Image.new("RGB", img_raw.size, (255, 255, 255)) 
            background.paste(img_raw, mask=img_raw.split()[3]) # 3 is the alpha channel
            img = background
        else:
            img = img_raw.convert("RGB")

        # Apply transforms
        image_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Perform inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            class_index = outputs.argmax(dim=1).item()
            food_prob = probabilities[0, class_index].item()
            
        is_food_pred = food_prob >= self.confidence_threshold
        
        if show_plot:
            plt.figure(figsize=(6, 6))
            plt.imshow(img_raw)
            pred_label = 'Food' if is_food_pred else 'Not Food'
            plt.title(f"Prediction: {pred_label} (Prob: {food_prob:.4f})")
            plt.axis('off')
            plt.show()

        return is_food_pred

    def get_food_array(self, folder_path: str) -> np.ndarray:
        """
        Evaluates all images in a folder and returns a NumPy array of 0s and 1s,
        where 1 indicates the image contains food, and 0 otherwise. 
        
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
            list: A list of NumPy arrays (one for each subfolder) containing 0s and 1s.
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
        
        Args:
            input_folder_path (str): path to the folder containing subfolders of files.
            food_arrays (list): list of 1D arrays/lists (0s and 1s) representing food or not.
            output_folder_path (str): path where the filtered subfolders and their files will be saved.
        """
        if not os.path.exists(input_folder_path):
            raise FileNotFoundError(f"Input folder not found: {input_folder_path}")
            
        os.makedirs(output_folder_path, exist_ok=True)
        
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
            
            os.makedirs(output_subfolder_path, exist_ok=True)
            
            files = [f for f in os.listdir(input_subfolder_path) 
                     if os.path.isfile(os.path.join(input_subfolder_path, f))]
            files = sorted(files)
            
            food_array = food_arrays[i]
            
            for j, file_name in enumerate(files):
                if j >= len(food_array):
                    break
                    
                if food_array[j] == 1:
                    src_file = os.path.join(input_subfolder_path, file_name)
                    dst_file = os.path.join(output_subfolder_path, file_name)
                    shutil.copy2(src_file, dst_file)
