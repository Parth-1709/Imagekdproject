import os
import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from student_model import StudentCNN
 # Make sure your model class is here

# Paths
blurry_dir = 'data/blurry'
output_dir = 'student_outputs'
model_path = 'checkpoints/student_model.pth'

# Create output folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Image transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StudentCNN().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# Inference
image_names = sorted(os.listdir(blurry_dir))
with torch.no_grad():
    for name in image_names:
        path = os.path.join(blurry_dir, name)
        img = Image.open(path).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(device)

        output_tensor = model(input_tensor).squeeze(0).cpu()
        output_img = transforms.ToPILImage()(output_tensor.clamp(0, 1))
        output_img.save(os.path.join(output_dir, name))

print(f"Inference complete! Student outputs saved in: {output_dir}")
