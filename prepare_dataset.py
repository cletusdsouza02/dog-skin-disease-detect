import os
import sys
import shutil
import random

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def setup_splits():
    dataset_dir = "dataset"
    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")
    valid_dir = os.path.join(dataset_dir, "valid")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)

    # 1. Fix ringworm train folder
    train_ringworm = os.path.join(train_dir, "ringworm")
    test_ringworm = os.path.join(test_dir, "ringworm")
    os.makedirs(train_ringworm, exist_ok=True)

    train_rw_count = len(os.listdir(train_ringworm))
    test_rw_files = sorted(os.listdir(test_ringworm)) if os.path.exists(test_ringworm) else []

    if train_rw_count == 0 and len(test_rw_files) > 40:
        # Move 85 files to train, keep the rest in test
        move_to_train = test_rw_files[:85]
        for f in move_to_train:
            src = os.path.join(test_ringworm, f)
            dst = os.path.join(train_ringworm, f)
            shutil.move(src, dst)
        print(f"[✓] Moved {len(move_to_train)} ringworm images to {train_ringworm}")

    # 2. Populate valid split from train (15% of train) if valid is empty
    classes = [
        "Dermatitis",
        "Fungal_infections",
        "Healthy",
        "Hypersensitivity",
        "demodicosis",
        "ringworm"
    ]

    for c in classes:
        c_train = os.path.join(train_dir, c)
        c_val = os.path.join(valid_dir, c)
        os.makedirs(c_val, exist_ok=True)

        val_files = os.listdir(c_val)
        train_files = os.listdir(c_train) if os.path.exists(c_train) else []

        if len(val_files) == 0 and len(train_files) > 10:
            num_val = max(5, int(len(train_files) * 0.15))
            random.seed(42)
            chosen_val = random.sample(train_files, num_val)
            for f in chosen_val:
                src = os.path.join(c_train, f)
                dst = os.path.join(c_val, f)
                shutil.move(src, dst)
            print(f"[✓] Created validation split for {c}: {num_val} images")

    print("\n=== Dataset Distribution Summary ===")
    for split in ["train", "valid", "test"]:
        sp_path = os.path.join(dataset_dir, split)
        counts = {}
        total = 0
        for c in classes:
            cp = os.path.join(sp_path, c)
            cnt = len(os.listdir(cp)) if os.path.exists(cp) else 0
            counts[c] = cnt
            total += cnt
        print(f"Split '{split}' (Total: {total}): {counts}")

if __name__ == "__main__":
    setup_splits()
