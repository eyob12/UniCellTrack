
# 🧬 UniCellTrack

A unified framework for segmentation and tracking of densely packed cells in 2D time-lapse microscopy.
<p align="center">
  <img src="https://github.com/eyob12/UniCellTrack/blob/main/Result/Fhela_tracking_result.gif?raw=true" width="45%" />
  <img src="https://github.com/eyob12/UniCellTrack/blob/main/Result/MUSC_tracking_result.gif?raw=true" width="45%" />
</p>

---


## 📜 Code Availability

The source code for this repository will be publicly released upon the acceptance of the associated research paper. Stay tuned for updates!

---

## 🧩 Overview of Methodology

![Overview](https://github.com/user-attachments/assets/abd3f2ac-f023-4d83-ab31-efa876fc46cc)

---

## ✨ Key Features

* 🧬 **Dataset Support**: Tested on nine 2D datasets from the [Cell Tracking Challenge (CTC)](http://celltrackingchallenge.net/).
* 🧠 **SAM-based Segmentation**: Boundary aware SAM Adaption for multi-modality microscopy.
* 📌 **Self-Prompting**: Automatically generates point prompts from distance maps via DBSCAN.
* 🔁 **Hybrid Tracking**: Graph-based object linking + optical flow for robust temporal association.
* 🎨 **Visualizations**: Interactive overlay of segmentation masks, distance maps, and tracking results.
* 💾 **Export Support**: Saves instance masks, tracking labels, and tracking `.txt` in CTC format.

---

## 📂 Supported Datasets

* `BF-C2DL-HSC`, `BF-C2DL-MuSC`, `Fluo-C2DL-MSC`, `Fluo-N2DH-GOWT1`, `Fluo-N2DL-HeLa`, `PhC-C2DL-PSC`, `PhC-C2DH-U373`, `DIC-C2DH-HeLa`, `Fluo-N2DH-SIM+`

Each dataset is handled in the CTC format with frame-wise `.tif` images and corresponding `GT` masks.

---

## 📊 Results

Quantitative tracking and segmentation performance across all CTC benchmarks (see paper for detailed metrics and visualizations).
Performance highlights include high TRA scores and reduced manual corrections, demonstrating the value of prompt-free segmentation and hybrid temporal linking.

---

## 🗂️ Repository Structure

```
UniCellTrack/
├── preprocessing/          # Preprocessing scripts for distance maps, masks, resizing
├── training.py             # Training pipeline with SAM + distance map loss
├── test.py                 # Inference pipeline for segmentation + tracking
├── saved_model/            # Checkpoints or converted SAM weights
├── evaluation/             # Scripts for metric computation and result generation
├── datasets/               # Downloaded or linked datasets (CTC-style folder layout)
├── utils/                  # Prompting, post-processing, optical flow, graph utils
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/eyob12/UniCellTrack.git
cd UniCellTrack
pip install -r requirements.txt
```

---

## 🚀 Usage

### Dataset Preparation

Place your CTC datasets in the `datasets/` directory with the following structure:

```
datasets/
└── Fluo-N2DL-HeLa/
    ├── 01/
    │   ├── t000.tif
    │   ├── t001.tif
    │   └── ...
    └── 01_GT/
        └── SEG/
            ├── man_seg000.tif
            └── ...
```

### Preprocessing

```bash
python preprocessing/generate_distance_maps.py
```

### Train the Model

```bash
python training.py --dataset_dir datasets/ --output_dir saved_model/
```

### Run Inference & Tracking

```bash
python test.py --model_path saved_model/checkpoint.pth --dataset Fluo-N2DL-HeLa
```

---

## 📈 Evaluation

Tracking results are saved in `res_track.txt` format compatible with the CTC evaluation tool.
Segmentation outputs are stored as labeled masks and optionally visualized with overlays.

---

## 📖 Citation

If you use UniCellTrack in your research, please cite:

```

```


