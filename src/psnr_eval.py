import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image
import numpy as np

sharp_dir = 'data/sharp'
student_dir = 'data/student_output'

sharp_images = sorted(os.listdir(sharp_dir))
student_images = sorted(os.listdir(student_dir))

total_psnr = 0.0
count = 0

for s_img, st_img in zip(sharp_images, student_images):
    sharp_path = os.path.join(sharp_dir, s_img)
    student_path = os.path.join(student_dir, st_img)

    if not os.path.exists(sharp_path) or not os.path.exists(student_path):
        print(f"Skipping {s_img} (missing image)")
        continue

    sharp = np.array(Image.open(sharp_path).convert('RGB'))
    student = np.array(Image.open(student_path).convert('RGB'))

    # Resize if mismatched
    if sharp.shape != student.shape:
        student = np.array(Image.fromarray(student).resize((sharp.shape[1], sharp.shape[0])))

    score = psnr(sharp, student, data_range=255)
    total_psnr += score
    count += 1

avg_psnr = total_psnr / count
print(f"\n✅ Average PSNR over {count} images: {avg_psnr:.2f} dB")
