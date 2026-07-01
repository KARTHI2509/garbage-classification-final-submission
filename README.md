# Garbage Classification using MobileNetV3Large

This project implements a Deep Learning model to classify garbage into 6 categories: cardboard, glass, metal, paper, plastic, and trash. It utilizes transfer learning with a pre-trained **MobileNetV3Large** network and provides a web-based interactive GUI using **Gradio**.

---

## 🛠️ Project Structure
```
garbage-classification-final-submission/
├── garbage-classification-final-submission-main/
│   ├── finalsubmission.ipynb         # Model Training & Inference Notebook
│   ├── garbage classification.pptx    # Project Presentation
│   └── README.md                     # Project documentation (this file)
└── requirements.txt                  # Python dependencies
```

---

## 🚀 Key Improvements & Bug Fixes

A full code review has been performed on the model pipeline. Here are the recommended changes to fix logical and runtime bugs in the training/inference pipeline:

### 1. Model Data Leakage
* **Issue**: During training (`model.fit`), `validation_data` was set to the entire validation set (`val_ds`). However, the test dataset (`test_ds`) was derived from `val_ds` via `val_ds.take()`. As a result, the model was exposed to test data during the `EarlyStopping` callback.
* **Fix**: Change `validation_data=val_ds` to `validation_data=val_dat` (the partition of the validation set that excludes the test set).

### 2. Preprocessing Mismatch in Inference
* **Issue**: The `MobileNetV3Large` model is defined with `include_preprocessing=True`, embedding a rescaling layer internally. During training, raw float images `[0, 255]` are fed. However, in `app.py` / Gradio prediction, manual `preprocess_input` is called, scaling input to `[-1, 1]` which then gets rescaled *again* internally, causing extremely poor inference accuracy.
* **Fix**: Remove `img_array = preprocess_input(img_array)` from the inference prediction code.

---

## 📦 Getting Started

### 1. Installation
Ensure you have Python 3.10+ installed. Clone the repository and run:
```bash
pip install -r requirements.txt
```

### 2. Dataset
Make sure to download the dataset and place it in the folder matching the path config or customize `dataset_dir` in `finalsubmission.ipynb`.

### 3. Running the Jupyter Notebook
Open the notebook and run all cells:
```bash
jupyter notebook garbage-classification-final-submission-main/finalsubmission.ipynb
```

### 4. Running the Web App GUI
The final cells in the notebook will launch a Gradio interface. Open the local address printed (typically `http://127.0.0.1:7860`) in your browser to test classifications interactively.