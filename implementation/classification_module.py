import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm


class FoodClassifier:
    """
    A classifier module for food image classification using Vision Transformer.
    """

    def __init__(self):
        """
        Initialize the FoodClassifier.
        """
        self.model_path = "saved_models/model_1_vit_segment_aware.pth"
        self.labels_path = "meta/labels.txt"
        self.num_classes = 19
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load labels
        self.labels = self._load_labels()

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

    def _load_labels(self):
        """
        Load the class labels from the labels file.

        Returns:
            list: List of class names.
        """
        if not os.path.exists(self.labels_path):
            raise FileNotFoundError(f"Labels file not found at {self.labels_path}")
            
        with open(self.labels_path, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]
            
        if len(labels) != self.num_classes:
            print(f"Warning: Expected {self.num_classes} labels, found {len(labels)}")
            
        return labels

    def _load_model(self):
        """
        Load the model from checkpoint.

        Returns:
            torch.nn.Module: Loaded model
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

    def classify_by_image(self, image):
        """
        Classify a food image directly from PIL Image object.

        Args:
            image (PIL.Image): PIL Image object

        Returns:
            dict: Dictionary containing:
                  - 'class_index': Predicted class index
                  - 'class_name': Predicted class name
                  - 'confidence': Confidence score (softmax probability)
        """
        if not isinstance(image, Image.Image):
            raise TypeError("Input must be a PIL Image object")

        # Apply transforms
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Perform inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            class_index = outputs.argmax(dim=1).item()
            confidence = probabilities[0, class_index].item()
            
        class_name = self.labels[class_index] if class_index < len(self.labels) else f"Unknown ({class_index})"

        return {
            'class_index': class_index,
            'class_name': class_name,
            'confidence': confidence
        }

    def classify_by_path(self, image_path):
        """
        Classify a food image from file path.

        Args:
            image_path (str): Path to the image file

        Returns:
            dict: Dictionary containing:
                  - 'class_index': Predicted class index
                  - 'class_name': Predicted class name
                  - 'confidence': Confidence score (softmax probability)
                  - 'image_path': Original image path
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found at {image_path}")

        # Open image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"Failed to open image: {e}")

        # Classify using image
        result = self.classify_by_image(image)
        result['image_path'] = image_path

        return result
