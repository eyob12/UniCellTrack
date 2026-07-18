
# 🧬 UniCellTrack

This is an official repo for UniCellTrack paper .  ["A unified segmentation model with motion-aware graph matching for cross-modality cell tracking"](https://doi.org/10.1016/j.bspc.2026.110729/)
<div align="center">
  <!-- Top image (centered) -->
  <img src="https://github.com/eyob12/UniCellTrack/blob/main/Result/Fhela_tracking_result.gif?raw=true" width="90%" />
  
  <!-- Bottom row images with space to align like a T -->
  <table>
    <tr>
      <td align="right" width="45%">
        <img src="https://github.com/eyob12/UniCellTrack/blob/main/Result/MUSC_tracking_result.gif?raw=true" width="90%" />
      </td>
      <td width="10%"></td> <!-- spacer column -->
      <td align="left" width="45%">
        <img src="https://github.com/eyob12/UniCellTrack/blob/main/Result/GOWT1_tracking_result.gif?raw=true" width="90%" />
      </td>
    </tr>
  </table>
</div>




---



## 🧩 Overview of Methodology


<img width="8002" height="6427" alt="final_CT4" src="https://github.com/user-attachments/assets/ec5eb822-c0f8-46f6-93cd-d8c922509b37" />

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
python main.py
```

### Run Inference & Tracking

```bash
python tracking.py 
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


