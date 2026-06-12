# HUIR: Harnessing the Reciprocal Influence of Social Connections and User Interactions for Recommendation
## Requirements
```text
python = 3.10.18
torch = 2.2.0
torch-geometric = 2.3.1
numpy = 1.26.4
scipy = 1.15.3
pandas = 2.3.0
tqdm = 4.67.1
scikit-learn = 1.4.0
```

## 📌 Overview
The overview of HUIR:
![overview](image_static/overview.png)

The overview of HUIR:
![overview](image_static/fig2.png)

The overview of HUIR:
![overview](image_static/fig3.png)

## Datasets
We evaluate HUIR on three public multimodal recommendation datasets: Baby, Sports, and Clothing.

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

The recommended hyperparameters for ciao, douban, and flickr are provided in:

```text
src/configs/model/PA2PD.yaml
```

Before training, please open `src/configs/model/PA2PD.yaml`, keep only the target dataset configuration active, and comment out the other two dataset configurations.

### ciao

```bash
python main.py --dataset='ciao' --checkpoint='./Model/ciao/_tem_.pth' --model_dir='./Model/ciao/' --lr=0.005 --difflr=0.001 --decay=0.985 --reg=0.01 --noise_min=0.0001 --noise_max=0.1 --SRPCloss=0.05 --ISECloss=1e-3 --bprloss=2 --s_layers=4
```

### douban

```bash
python main.py --dataset='douban' --checkpoint='./Model/douban/_tem_.pth' --model_dir='./Model/douban/' --lr=5e-4 --difflr=5e-4 --decay=0.98 --reg=1e-8 --noise_max=0.01 --SRPCloss=0.01 --ISECloss=1e-4 --bprloss=1.5
```

### flickr

```bash
python main.py --dataset='flickr' --checkpoint='./Model/flickr/_tem_.pth' --model_dir=  './Model/flickr/' --difflr=5e-4 --noise_min=1e-6 --noise_max=0.05 --SRPCloss=0.5 --ISECloss=5e-3 --bprloss=10
```


## Acknowledgement

This code is developed based on the implementation of [MENTOR](https://github.com/Jinfeng-Xu/MENTOR). We sincerely thank the authors of MENTOR for their excellent work and for making their code publicly available, which provides an important foundation for this project.
