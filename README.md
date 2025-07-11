# 📷 Image Sharpening using Knowledge Distillation

A deep learning project that performs real-time image sharpening using a **Knowledge Distillation** framework. Designed for applications like video conferencing where low-quality streams are common, the lightweight student model mimics a high-performance teacher model to sharpen blurry inputs efficiently.

---

## 📌 Problem Statement

**Objective**:  
To enhance blurry images using a lightweight CNN model trained via knowledge distillation from a heavy-weight transformer model (Restormer), optimized for real-time performance.

---

## 🧠 Knowledge Distillation Framework

- **Teacher Model**: [Restormer](https://arxiv.org/abs/2111.09881) — pretrained, transformer-based model for image deblurring.
- **Student Model**: Custom-built 3-layer CNN trained on outputs (soft labels) from the teacher.

The student model achieves comparable sharpening results at **~387 FPS**, making it suitable for real-time use cases.

---

## 🗂 Dataset

📥 **Full Dataset (8 GB)** hosted on Google Drive:  
🔗 [Download from here](https://drive.google.com/drive/folders/1js63_lLxa3rbLCz4LT6j4WzeH6oQ7qw-?usp=drive_link)

**Contents:**
- `blurry/` — Degraded input images (bicubic downscaling)
- `sharp/` — Ground truth high-resolution images
- `student_output/` — Model predictions (student)
- `teacher_output/` — Model predictions (teacher)

> ✅ A sample version is included in the repository under `/SampleData/` for demo/testing.

---

## 🏗️ Project Structure
imagesharpening/
├── checkpoints/ # Saved student model
├── src/
│ ├── dataset.py
│ ├── train_student.py
│ ├── test_single_image.py
│ └── metrics/
│ ├── psnr_test.py
│ ├── ssim_test.py
│ └── inference_time.py
├── SampleData/ # Sample blurred, sharp, and output images
├── Restormer/ # External teacher model repo (ignored in .git)
└── Image Sharpening Report.docx


---

## 🧪 Training & Evaluation

- **Epochs**: 10  
- **Batch Size**: 8  
- **Optimizer**: Adam  
- **Loss Function**: MSE Loss  

| Metric                          | Student Model | Teacher Model |
| ------------------------------ | ------------- | ------------- |
| **SSIM**                        | 0.9320        | 0.9591        |
| **PSNR**                        | 22.37 dB      | 23.62 dB      |
| **FPS**                         | 387           | 3.5           |
| **Inference Time (900 images)**| 2.3 seconds   | 252 seconds   |

> 🔍 **Interpretation**: Teacher model may underperform slightly in PSNR due to higher sharpening aggressiveness, especially on images with strong Gaussian blur. Student preserves balance between speed and perceptual quality.


## 🧑‍💻 Setup Instructions

### ✅ Clone the repo

-git clone https://github.com/Parth-1709/Imagekdproject.git
cd Imagekdproject

---

 Install requirements
bash
Copy
Edit
pip install -r requirements.txt

---

📄 References
Restormer Paper: https://arxiv.org/abs/2111.09881

DIV2K Dataset: https://data.vision.ee.ethz.ch/cvl/DIV2K/

PyTorch Documentation: https://pytorch.org/

Knowledge Distillation: Hinton et al., 2015 — https://arxiv.org/abs/1503.02531

---

👨‍🎓 Author
Parth Prakash
📧 parth.prakash017@gmail.com 
🔗 GitHub Profile

