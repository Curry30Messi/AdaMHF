# AdaMHF
Adaptive Multi-modal Hierarchical Fusion for Survival Analysis

![Model Architecture](./model.png)

## Overview
AdaMHF (Adaptive Multimodal Hierarchical Fusion for Survival Prediction) is a deep learning framework designed for survival analysis using multi-modal data from whole slide images (WSIs) and genomic profiles. The framework adaptively fuses features from different modalities to improve survival prediction accuracy.

More details to be added.

## Installation
### Prerequisites
```bash
Python >= 3.7
PyTorch >= 1.12.1+cu113
scikit-survival >= 0.19.0
```

### Data Preparation
#### Whole Slide Images (WSIs)
1. Download WSIs from [TCGA](https://portal.gdc.cancer.gov/)
2. Extract features using the WSI processing pipeline from [CLAM](https://github.com/mahmoodlab/CLAM)
3. Save extracted features as .pt files for each WSI

Required data structure:
```bash
DATA_ROOT_DIR/
    └──pt_files/
        ├── slide1.pt
        ├── slide2.pt
        └── ...
```
Note: Specify DATA_ROOT_DIR using the --data_root_dir argument in run.sh

#### Genomic Data
The framework uses genomic profiles provided by [PIBD](https://github.com/zylbuaa/PIBD). Store these profiles in the csv/ directory.

## Cross-Validation
The model employs 5-fold cross-validation. Prepare the splits in advance and store them in the `splits/5foldcv` directory.

## Training
To train the model:
```bash
bash run.sh
```
Customize training parameters by modifying run.sh

## Model Performance
[ To be added ]

## Citation
If you find this work useful, please cite our paper:
```
[Citation details to be added]
```

## Acknowledgements
We thank the authors of the following open-source projects:
- [CLAM](https://github.com/mahmoodlab/CLAM) - For WSI processing pipeline.
- [PIBD](https://github.com/zylbuaa/PIBD) - For genomic profile processing.
- [CMTA](https://github.com/FT-ZHOU-ZZZ/CMTA) - For multi-modal learning insights and code base.


## Contact
For questions and issues, please open an issue in this repository.

