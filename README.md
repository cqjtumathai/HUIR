# PA²PD: Pseudo Adversarial Alignment and Preference Decorrelation Model for Multimodal Recommendation
## Requirements
```text
python = 3.8.20
pytorch = 2.1.2
torch-geometric = 2.6.1
numpy = 1.24.2
scipy = 1.10.1
pandas = 1.5.3
tqdm = 4.64.1
scikit-learn = 1.3.2
```
## Datasets
We evaluate PA²PD on three public multimodal recommendation datasets: Baby, Sports, and Clothing.

The processed datasets and pre-extracted multimodal features can be downloaded from:
- [Baby/Sports/Clothing](https://drive.google.com/drive/folders/1tU4IxYbLXMkp_DbIOPGvCry16uPvolLk)
Please place the downloaded datasets under the `data/` directory:

```text
data/
├── baby/
├── sports/
└── clothing/
```

| Dataset | # Users | # Items | # Interactions | Sparsity | Modality |
|:--|--:|--:|--:|--:|:--|
| Baby | 19,445 | 7,050 | 160,792 | 99.88% | Visual, Textual |
| Sports | 35,598 | 18,357 | 296,337 | 99.95% | Visual, Textual |
| Clothing | 39,387 | 23,033 | 278,677 | 99.97% | Visual, Textual |

Following previous multimodal recommendation studies, we use publicly released pre-extracted multimodal features:

- Visual feature dimension: `4096`
- Textual feature dimension: `384`


## Training

The recommended hyperparameters for Baby, Sports, and Clothing are provided in:

```text
src/configs/model/PA2PD.yaml
```

Before training, please open `src/configs/model/PA2PD.yaml`, keep only the target dataset configuration active, and comment out the other two dataset configurations.

### Baby

```bash
python main.py -m PA2PD -d baby
```

### Sports

```bash
python main.py -m PA2PD -d sports
```

### Clothing

```bash
python main.py -m PA2PD -d clothing
```

Please make sure that the dataset specified by `-d` is consistent with the active configuration in `src/configs/model/PA2PD.yaml`.


## Acknowledgement

This code is developed based on the implementation of [MENTOR](https://github.com/Jinfeng-Xu/MENTOR). We sincerely thank the authors of MENTOR for their excellent work and for making their code publicly available, which provides an important foundation for this project.
