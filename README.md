# AdaMHF


### Pre-requisites:
```bash
torch 1.12.1+cu113
scikit-survival 0.19.0
```
### Prepare your data
#### WSIs
1. Download WSIs from [TCGA](https://portal.gdc.cancer.gov/)
2. Use the WSI processing tool provided by [CLAM](https://github.com/mahmoodlab/CLAM) to extract features.

The structure of WSIs data should be as following:
```bash
DATA_ROOT_DIR/
    └──pt_files/
        ├── slide1.pt
        ├── slide2.pt
        └── ...
```

#### Genomics
In this AdaMHF, we directly use the genomic profiles provided by [PIBD](https://github.com/zylbuaa/PIBD), which should be stored in folder csv.

## Training-Validation Splits
We employ five-fold cross-validation, for which the splits should be prepared in advance and placed in the `splits/5foldcv` folder.



## Running Experiments
To train AdaMHF, you can specify the argument in the bash `run.sh` and run the command:
```bash
bash run.sh
```
## Acknowledgements
Huge thanks to the authors of following open-source projects:
- [CLAM](https://github.com/mahmoodlab/CLAM)
- [PIBD](https://github.com/zylbuaa/PIBD)
- [CMTA](https://github.com/FT-ZHOU-ZZZ/CMTA)
# The complete README file will be released shortly.

