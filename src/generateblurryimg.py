import os
import cv2

# Get the current script's directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define paths
sharp_dir = os.path.join(base_dir, "../data/sharp")
blurry_dir = os.path.join(base_dir, "../data/blurry")

# Create the blurry output folder if it doesn't exist
os.makedirs(blurry_dir, exist_ok=True)

# Loop through all images in sharp_dir
for filename in os.listdir(sharp_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Check for image files
        img_path = os.path.join(sharp_dir, filename)
        image = cv2.imread(img_path)

        if image is None:
            print(f"❌ Skipped unreadable image: {filename}")
            continue

        # Apply Gaussian Blur
        blurred_image = cv2.GaussianBlur(image, (21, 21), 5)


        # Save the blurred image
        save_path = os.path.join(blurry_dir, filename)
        cv2.imwrite(save_path, blurred_image)

        print(f"✅ Blurred image saved: {filename}")
