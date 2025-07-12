# 📷 Image Sharpening using Knowledge Distillation

A deep learning project that performs real-time image sharpening using a **Knowledge Distillation** framework. Designed for applications like video conferencing where low-quality streams are common, the lightweight student model mimics a high-performance teacher model to sharpen blurry inputs efficiently.

---

## 📌 Problem Statement

**Objective**:  
To enhance blurry images using a lightweight CNN model trained via knowledge distillation from a heavy-weight transformer model (Restormer), optimized for real-time performance.

---

## 🧠 Knowledge Distillation Framework

- **Teacher Model**: [Restormer](https://github.com/swz30/Restormer) — pretrained, transformer-based model for image deblurring.
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

```
imagesharpening/
├── checkpoints/                 # Saved student model
├── src/
│   ├── dataset.py              # Dataset loader
│   ├── train_student.py       # Student training script
│   ├── test_single_image.py   # Inference on single image
│   └── metrics/               # Evaluation metrics
│       ├── psnr_test.py
│       ├── ssim_test.py
│       └── inference_time.py
├── SampleData/                 # Sample blurred, sharp, and output images
├── Restormer/                  # Teacher model repo (ignored in git)
└── Image Sharpening Report.docx
```



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


## 🚀 Setup Instructions

### ✅ Prerequisites

- Python 3.8+
- pip
- Git
- (Optional) CUDA-compatible GPU for training

---

### 📦 Clone the Repository

```bash
git clone https://github.com/your-username/imagesharpening.git
cd imagesharpening

🧪 Create & Activate Virtual Environment
Copy
Edit
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

📚 Install Dependencies
Copy
Edit
pip install -r requirements.txt

Or install manually:
Copy
Edit
pip install torch torchvision opencv-python numpy matplotlib

📥 Download the Teacher Model (Restormer)
bash
Copy
Edit
git clone https://github.com/swz30/Restormer.git
```
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
🔗 GitHub Profile -> https://github.com/Parth-1709

---

MIT License

Copyright (c) 2025 Parth Prakash

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


