# radioanalyze V2 – Knee Cartilage Segmentation (MONAI + 3D UNet)

This repo is a scriptified version of the original Colab notebook `radioalayze V2.ipynb`.  
It trains and evaluates a 3D MONAI UNet for femoral and tibial cartilage segmentation on the OAIZIB-CM knee MRI dataset.

## Features

- HuggingFace OAIZIB-CM download (`load_dataset-support` branch)
- Automatic unzip + canonical folder layout (`imagesTr`, `labelsTr`, `imagesTs`, `labelsTs`)
- 3D UNet with Focal Tversky loss
- Heavy 3D augmentations + foreground-biased cropping
- Sliding-window inference with test-time augmentation (axis flips)
- Largest-connected-component post-processing per class
- Test Dice metrics (femur, tibia, combined)
- QC export: NIfTI prediction + MP4 overlay for first test case

## Install

```bash
git clone <your-repo-url>.git
cd radioanalyze-knee
pip install -r requirements.txt
python train_radioanalyze_v2.py --download-data --run-qc

# (Optional) train first if you haven't yet
python train_radioanalyze_v2.py --download-data

# Run inference on a new case
python infer_new_case.py \
  --input-path /path/to/dicom_or_nii \
  --weights-path ./weights/monai_knee_ft_best.pth \
  --out-pred ./new_case_seg.nii.gz
