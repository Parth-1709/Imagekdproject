from skimage.metrics import structural_similarity as ssim
from skimage.io import imread
from skimage.transform import resize
import os
import numpy as np

# Paths
sharp_folder = "data/sharp"
student_folder = "data/student_output"
teacher_folder = "data/teacher_output"

sharp_images = sorted(os.listdir(sharp_folder))
student_images = sorted(os.listdir(student_folder))
teacher_images = sorted(os.listdir(teacher_folder))

ssim_student_total = 0.0
ssim_teacher_total = 0.0
count = 0

for img_name in sharp_images:
    sharp_path = os.path.join(sharp_folder, img_name)
    student_path = os.path.join(student_folder, img_name)
    teacher_path = os.path.join(teacher_folder, img_name)

    if not os.path.exists(student_path) or not os.path.exists(teacher_path):
        print(f"Missing corresponding image for: {img_name}")
        continue

    sharp_img = imread(sharp_path)
    student_img = imread(student_path)
    teacher_img = imread(teacher_path)

    # Resize to match if needed
    target_shape = (256, 256, 3)
    if sharp_img.shape != target_shape:
        sharp_img = resize(sharp_img, target_shape, preserve_range=True, anti_aliasing=True).astype(np.uint8)
    if student_img.shape != target_shape:
        student_img = resize(student_img, target_shape, preserve_range=True, anti_aliasing=True).astype(np.uint8)
    if teacher_img.shape != target_shape:
        teacher_img = resize(teacher_img, target_shape, preserve_range=True, anti_aliasing=True).astype(np.uint8)

    try:
        ssim_student = ssim(sharp_img, student_img, channel_axis=-1)
        ssim_teacher = ssim(sharp_img, teacher_img, channel_axis=-1)
    except Exception as e:
        print(f"SSIM error on {img_name}: {e}")
        continue

    ssim_student_total += ssim_student
    ssim_teacher_total += ssim_teacher
    count += 1

if count > 0:
    print(f"Average SSIM (Student): {ssim_student_total / count:.4f}")
    print(f"Average SSIM (Teacher): {ssim_teacher_total / count:.4f}")
else:
    print("No valid image pairs found. Check image folder structure or dimensions.")
