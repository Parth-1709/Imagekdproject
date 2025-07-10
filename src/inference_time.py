import os
import time
from PIL import Image
from torchvision import transforms
import torch
from student_model import StudentCNN
 # replace with your student model class
from restormer_runner import get_restormer  # you'll define this function to load Restormer

def load_images(img_dir):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    images = []
    for img_name in sorted(os.listdir(img_dir)):
        img_path = os.path.join(img_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        img = transform(img).unsqueeze(0)
        images.append(img)
    return images

def time_model(model, images, device):
    model.eval()
    model.to(device)
    with torch.no_grad():
        start = time.time()
        for img in images:
            img = img.to(device)
            _ = model(img)
        end = time.time()
    total_time = end - start
    fps = len(images) / total_time
    return total_time, fps

if __name__ == "__main__":
    img_dir = "data/blurry"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = load_images(img_dir[:50])  # time only a few to avoid long test

    # Student model
    student = StudentCNN()
    student.load_state_dict(torch.load("checkpoints/student_model.pth"))
    total_time, fps = time_model(student, images, device)
    print(f"Student: {total_time:.2f}s, FPS: {fps:.2f}")

    # Teacher model
    restormer = get_restormer(device)
    total_time, fps = time_model(restormer, images, device)
    print(f"Teacher: {total_time:.2f}s, FPS: {fps:.2f}")
