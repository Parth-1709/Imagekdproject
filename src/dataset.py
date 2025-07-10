import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ImageDeblurDataset(Dataset):
    def __init__(self, blurry_dir, teacher_dir, transform=None):
        self.blurry_dir = blurry_dir
        self.teacher_dir = teacher_dir
        self.transform = transform

        self.image_names = sorted(os.listdir(self.blurry_dir))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        blurry_path = os.path.join(self.blurry_dir, image_name)
        teacher_path = os.path.join(self.teacher_dir, image_name)

        blurry_img = Image.open(blurry_path).convert('RGB')
        teacher_img = Image.open(teacher_path).convert('RGB')

        if self.transform:
            blurry_img = self.transform(blurry_img)
            teacher_img = self.transform(teacher_img)

        return blurry_img, teacher_img


if __name__ == "__main__":
    blurry_path = "data/blurry"
    teacher_path = "data/teacher_output"

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = ImageDeblurDataset(blurry_path, teacher_path, transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    for i, (blurry, teacher_img) in enumerate(dataloader):
        print(f"Batch {i+1} - Blurry: {blurry.shape}, Teacher: {teacher_img.shape}")
        break
