# FB-CLIP
> [**CVPR 26**] [**FB-CLIP: Fine-Grained Zero-Shot Anomaly Detection with Foreground-Background Disentanglement**]





## Introduction 
Fine-grained anomaly detection is crucial in industrial and medical applications, but labeled anomalies are often scarce, making zero-shot detection challenging. While vision-language models like CLIP offer promising solutions, they struggle with foreground-background feature entanglement and coarse textual semantics.  We propose FB-CLIP, a framework that enhances anomaly localization via multi-strategy textual representations and foreground-background separation. In the textual modality, it combines End-of-Text features, global-pooled representations, and attention-weighted token features for richer semantic cues. In the visual modality, multi-view soft separation along identity, semantic, and spatial dimensions, together with background suppression, reduces interference and improves discriminability. Semantic Consistency Regularization (SCR) aligns image features with normal and abnormal textual prototypes, suppressing uncertain matches and enlarging semantic gaps.  Experiments show that FB-CLIP effectively distinguishes anomalies from complex backgrounds, achieving accurate fine-grained anomaly detection and localization under zero-shot settings. All experiments are conducted in PyTorch-2.0.0 with a single NVIDIA RTX 3090 24GB. 



## How to Run

```
conda create -n FBCLIP python=3.8.20
conda activate FBCLIP
pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Prepare your dataset
Following [AnomalyCLIP](https://github.com/zqhang/AnomalyCLIP) ,download the dataset below:

* Industrial Domain:
[MVTec](https://www.mvtec.com/company/research/datasets/mvtec-ad), [VisA](https://github.com/amazon-science/spot-diff), [MPDD](https://github.com/stepanje/MPDD), [BTAD](http://avires.dimi.uniud.it/papers/btad/btad.zip), [SDD](https://www.vicos.si/resources/kolektorsdd/), [DAGM](https://www.kaggle.com/datasets/mhskjelvareid/dagm-2007-competition-dataset-optical-inspection), [DTD-Synthetic](https://drive.google.com/drive/folders/10OyPzvI3H6llCZBxKxFlKWt1Pw1tkMK1)

* Medical Domain:
[HeadCT](https://drive.google.com/file/d/1lSAUkgZXUFwTqyexS8km4ZZ3hW89i5aS/view?usp=sharing), [BrainMRI](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection), [Br35H](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection), [ISIC](https://drive.google.com/file/d/1UeuKgF1QYfT1jTlYHjxKB3tRjrFHfFDR/view?usp=sharing), [CVC-ColonDB](https://figshare.com/articles/figure/Polyp_DataSet_zip/21221579), [CVC-ClinicDB](https://figshare.com/articles/figure/Polyp_DataSet_zip/21221579), [Kvasir](https://figshare.com/articles/figure/Polyp_DataSet_zip/21221579), [Endo](https://drive.google.com/file/d/1LNpLkv5ZlEUzr_RPN5rdOHaqk0SkZa3m/view), [TN3K](https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation?tab=readme-ov-file).

* Real-IAD dataset: [Real-IAD]( https://huggingface.co/datasets/Real-IAD/Real-IAD)


### Generate the dataset JSON
Take MVTec AD for example (With multiple anomaly categories)

Structure of MVTec Folder:
```
mvtec/
│
├── meta.json
│
├── bottle/
│   ├── ground_truth/
│   │   ├── broken_large/
│   │   │   └── 000_mask.png
|   |   |   └── ...
│   │   └── ...
│   └── test/
│       ├── broken_large/
│       │   └── 000.png
|       |   └── ...
│       └── ...
│   
└── ...
```

```bash
cd generate_dataset_json
python mvtec.py
```




### Run FB-CLIP
* Quick start (use the pre-trained weights)
```bash
bash test.sh
```
  
* Train your own weights
```bash
bash train_on_mvtec.sh
bash train_on_visa.sh
```

### To Test Real-IAD
Due to the large size of the Real-IAD dataset, we recommend testing on a per-category basis.
```bash
bash test_realAD_by_class.sh
```

* Our code is largely based on [AnomalyCLIP](https://github.com/zqhang/AnomalyCLIP) and [AF-CLIP](https://github.com/Faustinaqq/AF-CLIP). Thanks for these authors for their valuable work, hope our work can also contribute to related research.

Email: huming708@gmail.com


