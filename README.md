# DAAD

DAAD is an anomaly detection research codebase for CTCA image experiments. It was developed by extending the original SimpleNet implementation with CTCA-specific data splitting, image-level augmentation, feature-level augmentation, feature selection, cross-validation, and automatic evaluation result export.

This repository is based on:

- SimpleNet: A Simple Network for Image Anomaly Detection and Localization
- Original repository: https://github.com/DonaldRR/SimpleNet
- Paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_SimpleNet_A_Simple_Network_for_Image_Anomaly_Detection_and_Localization_CVPR_2023_paper.pdf

## Main Changes

Compared with the original SimpleNet code, this project adds:

- CTCA dataset loading with normal/disease patient folders.
- Normal-only 4-fold cross-validation.
- ILA, image-level augmentation for real and fake training images.
- FLA, feature-level augmentation using noise, interpolation, and extrapolation modes.
- FS, feature selection using GAP, GMP, both, or none.
- Automatic run-folder naming using CV fold, ILA, FLA, and FS settings.
- Automatic AUROC, AUPRC, AP, and inference-speed export to `evaluation_results.txt`.
- Batch experiment scripts for ablation, feature-selection ratio, image-gap, and backbone experiments.

## Dataset Layout

The expected CTCA dataset layout is:

```text
CTCA_8/
  test/
    N_1/
    N_2/
    ...
    N_20/
    D_1/
    D_2/
    ...
    D_20/
```

`N_*` folders are treated as normal cases and `D_*` folders are treated as disease cases.

For cross-validation, all disease cases are always used for testing. The 20 normal cases are split into 4 folds:

```text
Fold 1 test normals: N_1  - N_5
Fold 2 test normals: N_6  - N_10
Fold 3 test normals: N_11 - N_15
Fold 4 test normals: N_16 - N_20
```

For each fold, the remaining 15 normal cases are used as the unsupervised training set.

## Installation

Create and activate a Python environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

The code expects PyTorch with CUDA support when running GPU experiments.

## Single Experiment

Example main experiment:

```bash
python3 main.py \
  --gpu 1 \
  --seed 101 \
  --log_group daad_ctca_cv \
  --log_project CTCA \
  --results_path /home/jihun/PycharmProjects/DAAD/results \
  --run_name main_seed101 \
  net \
  -b resnet18 \
  -le layer2 \
  -le layer3 \
  --pretrain_embed_dimension 256 \
  --target_embed_dimension 256 \
  --patchsize 3 \
  --patchstride 2 \
  --meta_epochs 10 \
  --embedding_size 64 \
  --gan_epochs 1 \
  --noise_std 0.025 \
  --dsc_hidden 512 \
  --dsc_layers 2 \
  --dsc_margin .5 \
  --pre_proj 1 \
  --true_fla 5 \
  --false_fla 5 \
  --fs_mode both \
  --fs_keep_ratio 0.5 \
  --image_gap 5 \
  dataset \
  --cv_fold 1 \
  --cv_num_folds 4 \
  --real_aug 1 \
  --fake_aug 1 \
  --batch_size 1024 \
  --resize 48 \
  --imagesize 36 \
  --rotate_degrees 30 \
  --translate 0.2 \
  --brightness 0.2 \
  --scale 0.2 \
  --contrast 0.2 \
  --hflip 0.5 \
  --vflip 0.5 \
  CTCA /home/jihun/PycharmProjects/DAAD/CTCA_8
```

## Important Parameters

`--real_aug` and `--fake_aug` control image-level augmentation.

- `0`: disabled
- `1`: enabled

`--true_fla` and `--false_fla` control feature-level augmentation.

- `0`: original features
- `1`: original features with noise
- `2`: original features with interpolation
- `3`: original features with extrapolation
- `5`: interpolation and extrapolation
- `6`: noise, interpolation, and extrapolation

`--fs_mode` controls feature selection.

- `none`: disabled
- `gap`: GAP-based feature selection
- `gmp`: GMP-based feature selection
- `both`: union of GAP-selected and GMP-selected feature maps

`--fs_keep_ratio` controls the selection ratio for each FS mode.

`--image_gap` controls the gap used for feature-level interpolation and extrapolation.

## Batch CV Experiments

`연속 실험6` runs the current cross-validation experiment set. It includes:

- Main DAAD setting.
- ILA, FLA, and FS ablations.
- FS mode comparison.
- FS keep-ratio sensitivity.
- Image-gap sensitivity.
- Backbone comparison.

Run it with:

```bash
bash "연속 실험6"
```

The script has skip logic. If an experiment folder already contains `evaluation_results.txt`, that run is skipped.

## Outputs

The final output for each run is:

```text
results/CTCA/<log_group>/<run_name>/evaluation_results.txt
```

This file contains:

- AUROC
- AUPRC
- Average Precision
- Total test images
- Total test time
- Seconds per image
- Images per second
- Seconds per patient/dataset
- Group-wise abnormal-ratio statistics

Large generated files such as datasets, checkpoints, logs, Excel files, and plots are intentionally ignored by git.

## Notes

The implementation keeps `simplenet.py`, `SimpleNet`, and `MVTecDataset` as compatibility names because they are part of the original execution path. Project-facing documentation, logs, and experiment naming use DAAD.
