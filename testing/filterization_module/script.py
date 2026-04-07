import torch
import timm
from PIL import Image
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# 1. Configuration
MODEL_ID = "mrdbourke/food-not-food-classifier-csatv2-v1"
IMAGE_PATH = "/home/pranto/Documents/academics/Capstone/dataset/food_data_english/images/biriyani/000002.jpg"  # Change this to your image path

# 2. Load Model Architecture
# This model uses the 'csatv2.r512_in1k' backbone
model = timm.create_model("csatv2.r512_in1k", pretrained=False, num_classes=2)

# 3. Download and Load Weights
weights_path = hf_hub_download(repo_id=MODEL_ID, filename="model.safetensors")
state_dict = load_file(weights_path)
model.load_state_dict(state_dict)
model.eval()

# 4. Prepare Preprocessing
# Use timm's built-in data config resolver to get the exact transforms the model expects
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

# 5. Inference Function
def predict(img_path):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transforms(img).unsqueeze(0) # Add batch dimension

    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
    # Labels for this specific model:
    # 0: food_or_drink
    # 1: not_food_or_drink
    labels = ["Food/Drink", "Not Food"]
    confidences = {labels[i]: float(probabilities[i]) for i in range(len(labels))}
    
    top_label = labels[torch.argmax(probabilities)]
    return top_label, confidences

# Run it
label, score = predict(IMAGE_PATH)
print(f"Prediction: {label}")
print(f"Confidences: {score}")