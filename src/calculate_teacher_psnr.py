import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr

sharp_dir = "data/sharp"
teacher_dir = "data/teacher_output"

sharp_images = sorted(os.listdir(sharp_dir))
teacher_images = sorted(os.listdir(teacher_dir))

psnr_total = 0
count = 0

for sharp_img_name, teacher_img_name in zip(sharp_images, teacher_images):
    sharp_path = os.path.join(sharp_dir, sharp_img_name)
    teacher_path = os.path.join(teacher_dir, teacher_img_name)

    sharp_img = cv2.imread(sharp_path)
    teacher_img = cv2.imread(teacher_path)

    if sharp_img is None or teacher_img is None:
        print(f"Error reading image pair: {sharp_img_name}, {teacher_img_name}")
        continue

    psnr_val = psnr(sharp_img, teacher_img)
    psnr_total += psnr_val
    count += 1

average_psnr = psnr_total / count
print(f"Average PSNR between Teacher and Sharp: {average_psnr:.2f} dB")
