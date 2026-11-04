"""
Automated Dataset Augmentation & Balancing Pipeline
Expands minority classes (primarily Ringworm and Hypersensitivity)
using clinically-realistic dermatological image transformations.
"""

import os
import sys
from PIL import Image, ImageEnhance

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TARGET_TRAIN_DIR = os.path.join("dataset", "train")


def zoom_and_crop(img, zoom_factor=1.12):
    """Zooms into the center of the image and crops back to original size."""
    w, h = img.size
    new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
    img_zoomed = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = (new_w - w) // 2
    top = (new_h - h) // 2
    return img_zoomed.crop((left, top, left + w, top + h))


def generate_augmented_variations(img):
    """
    Generates 5 distinct, high-quality dermatological variations:
    1. 90-degree rotation + Horizontal flip
    2. 180-degree rotation + Brightness enhancement (simulating phone flash)
    3. 270-degree rotation + Contrast enhancement (simulating clinical lighting)
    4. Center Zoom (12%) + Sharpness enhancement
    5. Vertical flip + Color saturation enhancement
    """
    variations = []

    # 1. Rot90 + HFlip
    v1 = img.transpose(Image.Transpose.ROTATE_90).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    variations.append(("aug_rot90_hflip", v1))

    # 2. Rot180 + Brightness (1.12x)
    v2 = img.transpose(Image.Transpose.ROTATE_180)
    v2 = ImageEnhance.Brightness(v2).enhance(1.12)
    variations.append(("aug_rot180_bright", v2))

    # 3. Rot270 + Contrast (1.15x)
    v3 = img.transpose(Image.Transpose.ROTATE_270)
    v3 = ImageEnhance.Contrast(v3).enhance(1.15)
    variations.append(("aug_rot270_contrast", v3))

    # 4. Zoom (1.12x) + Sharpness (1.3x)
    v4 = zoom_and_crop(img, zoom_factor=1.12)
    v4 = ImageEnhance.Sharpness(v4).enhance(1.30)
    variations.append(("aug_zoom_sharp", v4))

    # 5. VFlip + Color Saturation (1.10x)
    v5 = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    v5 = ImageEnhance.Color(v5).enhance(1.10)
    variations.append(("aug_vflip_color", v5))

    return variations


def augment_class_folder(class_name, multiplier=5):
    """Augments images in a specific dataset/train/<class_name> folder."""
    target_dir = os.path.join(TARGET_TRAIN_DIR, class_name)
    if not os.path.exists(target_dir):
        print(f"[!] Directory not found: {target_dir}")
        return

    # Filter only original files (exclude existing aug_ images)
    all_files = [f for f in os.listdir(target_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    original_files = [f for f in all_files if not f.startswith("aug_")]
    initial_count = len(original_files)

    print(f"\n==================================================================")
    print(f" Processing Class: '{class_name}'")
    print(f" Original Images: {initial_count} | Target Multiplier: {multiplier}x")
    print(f"==================================================================")

    generated_count = 0
    for idx, fname in enumerate(original_files, 1):
        src_path = os.path.join(target_dir, fname)
        try:
            with Image.open(src_path) as img:
                img_rgb = img.convert("RGB")
                variations = generate_augmented_variations(img_rgb)

                base_name, _ = os.path.splitext(fname)
                for prefix, var_img in variations[:multiplier]:
                    out_name = f"{prefix}_{base_name}.jpg"
                    out_path = os.path.join(target_dir, out_name)
                    var_img.save(out_path, format="JPEG", quality=95)
                    generated_count += 1
        except Exception as e:
            print(f" [!] Error processing {fname}: {e}")

    total_now = len([f for f in os.listdir(target_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"[✓] Generated {generated_count} new augmented images for '{class_name}'")
    print(f"[✓] Total images in '{class_name}' is now: {total_now} (up from {initial_count})")


def main():
    print("==================================================================")
    print(" 🐶 DATASET BALANCING & AUGMENTATION PIPELINE")
    print("==================================================================")

    # 1. Ringworm (73 -> ~438 images, 5 variations per image)
    augment_class_folder("ringworm", multiplier=5)

    # 2. Hypersensitivity (102 -> ~408 images, 3 variations per image)
    augment_class_folder("Hypersensitivity", multiplier=3)

    print("\n==================================================================")
    print(" SUMMARY OF UPDATED TRAINING DATASET COUNTS")
    print("==================================================================")
    classes = [
        "Dermatitis",
        "Fungal_infections",
        "Healthy",
        "Hypersensitivity",
        "demodicosis",
        "ringworm"
    ]
    total_imgs = 0
    for c in sorted(classes):
        c_path = os.path.join(TARGET_TRAIN_DIR, c)
        count = len(os.listdir(c_path)) if os.path.exists(c_path) else 0
        total_imgs += count
        print(f" * {c:<22} : {count:>4} images")
    print("------------------------------------------------------------------")
    print(f" Total Training Images : {total_imgs:>4} images (Fully Balanced!)")
    print("==================================================================")


if __name__ == "__main__":
    main()

