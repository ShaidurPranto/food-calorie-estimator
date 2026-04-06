import torch
import timm
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

    def is_food(self, image_path: str) -> bool:
        """
        Predicts whether an image is food or not.
        
        Args:
            image_path (str): path to the image
            
        Returns:
            bool: True if probability of being food is > 0.5, False otherwise.
        """
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.transforms(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
        # Index 0 is food, Index 1 is not food
        food_prob = float(probabilities[0])
        return food_prob > 0.5
