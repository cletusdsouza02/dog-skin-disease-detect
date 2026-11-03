"""
Generates the standard dataset folder hierarchy for Dog Skin Disease Detection (6 Classes):
dataset/
  ├── train/ (6 classes)
  ├── valid/ (6 classes)
  └── test/  (6 classes)
"""

import os
from PIL import Image, ImageDraw

CLASSES = [
    "Dermatitis",
    "Fungal_infections",
    "Healthy",
    "Hypersensitivity",
    "demodicosis",
    "ringworm"
]

SPLITS = {
    "train": 8,
    "valid": 2,
    "test": 2
}

CLASS_COLORS = {
    "Dermatitis": (220, 60, 60),          # Red papules / pustules
    "Fungal_infections": (200, 140, 40),   # Hyperpigmented / yellowish
    "Healthy": (40, 180, 80),              # Clean supple pink/green
    "Hypersensitivity": (60, 120, 220),    # Allergic flare
    "demodicosis": (150, 70, 180),         # Crusty purplish grey
    "ringworm": (210, 120, 30)             # Scaly circular orange
}

def create_sample_image(class_name, img_idx, split_name):
    """Creates a 224x224 synthetic sample image representing the class."""
    img = Image.new("RGB", (224, 224), color=(240, 235, 230))
    draw = ImageDraw.Draw(img)
    
    base_color = CLASS_COLORS.get(class_name, (100, 100, 100))
    for offset in range(10, 80, 15):
        box = [
            50 + offset // 2 + (img_idx * 5) % 20,
            50 + offset // 2 + (img_idx * 7) % 20,
            174 - offset // 2,
            174 - offset // 2
        ]
        alpha_color = tuple(min(255, c + (offset * 2) % 40) for c in base_color)
        draw.ellipse(box, outline=alpha_color, width=3)
    
    draw.rectangle([10, 10, 214, 214], outline=(200, 200, 200), width=1)
    return img

def setup_dataset():
    base_dir = "dataset"
    print("==================================================================")
    print(" Verifying/Generating Dataset Directory Structure (6 Classes)...")
    print("==================================================================")
    
    total_created = 0
    for split, count in SPLITS.items():
        for cls in CLASSES:
            folder_path = os.path.join(base_dir, split, cls)
            os.makedirs(folder_path, exist_ok=True)
            
            for i in range(1, count + 1):
                img_path = os.path.join(folder_path, f"{cls}_{split}_{i:02d}.jpg")
                if not os.path.exists(img_path):
                    img = create_sample_image(cls, i, split)
                    img.save(img_path, "JPEG", quality=90)
                    total_created += 1
                    
    print(f"[OK] Dataset directory '{base_dir}/' verified for all {len(CLASSES)} classes!")
    print("==================================================================")

if __name__ == "__main__":
    setup_dataset()
