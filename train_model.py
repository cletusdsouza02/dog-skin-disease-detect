"""
Dog Skin Disease Detection - Training Pipeline Entrypoint
Allows running 'python train_model.py' directly.
Delegates to train_models.main().
"""

import sys
from train_models import (
    main,
    load_datasets,
    build_mobilenet_v2,
    build_efficientnet_b0,
    build_resnet50,
    compile_and_train,
    export_to_tflite,
    CLASS_NAMES,
    IMG_SIZE,
    BATCH_SIZE,
    NUM_CLASSES,
    EPOCHS
)

if __name__ == "__main__":
    main()
