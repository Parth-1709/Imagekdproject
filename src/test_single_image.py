import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from student_model import StudentCNN

# Load the image
img_path = img_path = r"C:\Users\Lenovo\OneDrive\画像\fix-blurry-photos.jpg"

input_img = Image.open(img_path).convert('RGB')

# Preprocess
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
input_tensor = transform(input_img).unsqueeze(0).to("cuda")

# Load model
model = StudentCNN().to("cuda")
model.load_state_dict(torch.load("checkpoints/student_model.pth"))
model.eval()

# Inference
with torch.no_grad():
    model_output = model(input_tensor)  # <-- this line defines model_output

# Convert output tensor to image
output_img = transforms.ToPILImage()(model_output.squeeze(0).cpu())

# Display
plt.subplot(1, 2, 1)
plt.title("Input Blurry")
plt.imshow(input_img.resize((256, 256)))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Output Sharp (Student)")
plt.imshow(output_img)
plt.axis('off')

plt.tight_layout()
plt.show()

