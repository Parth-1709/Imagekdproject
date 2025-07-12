import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import math
import numpy as np
from student_model import StudentCNN

# Load the high-resolution image
img_path = r"C:\Users\Lenovo\OneDrive\画像\fix-blurry-photos.jpg"
input_img = Image.open(img_path).convert('RGB')
orig_width, orig_height = input_img.size

# Set tile size and device
tile_size = 256
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define transforms (no resize)
to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

# Pad image to make it divisible by tile_size
pad_w = (tile_size - orig_width % tile_size) % tile_size
pad_h = (tile_size - orig_height % tile_size) % tile_size
padded_img = Image.new("RGB", (orig_width + pad_w, orig_height + pad_h))
padded_img.paste(input_img, (0, 0))

# Convert to tensor
padded_tensor = to_tensor(padded_img).unsqueeze(0).to(device)
_, _, padded_h, padded_w = padded_tensor.shape

# Load the student model
model = StudentCNN().to(device)
model.load_state_dict(torch.load("checkpoints/student_model.pth"))
model.eval()

# Output canvas
output_tensor = torch.zeros_like(padded_tensor)

# Process tiles
with torch.no_grad():
    for i in range(0, padded_h, tile_size):
        for j in range(0, padded_w, tile_size):
            tile = padded_tensor[:, :, i:i+tile_size, j:j+tile_size]
            output_tile = model(tile)
            output_tensor[:, :, i:i+tile_size, j:j+tile_size] = output_tile

# Crop output to original size
output_tensor = output_tensor[:, :, :orig_height, :orig_width]

# Convert tensors to images
output_img = to_pil(output_tensor.squeeze(0).cpu())
resized_input_img = input_img.resize((orig_width, orig_height))

# Display comparison
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Input Blurry")
plt.imshow(resized_input_img)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Output Sharp (Student)")
plt.imshow(output_img)
plt.axis("off")

plt.tight_layout()
plt.show()
