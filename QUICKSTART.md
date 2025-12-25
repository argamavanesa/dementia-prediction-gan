# 🧠 Dementia CGAN - Quick Start Guide

## ✅ Setup Complete!

App sudah siap dengan EXTREME VARIATION & download-only mode!

---

## 📂 Struktur Folder

```
dementia-prediction-gan/
├── app.py                    # Streamlit web app
├── inference.py              # CLI script
├── model_architecture.py     # Generator architecture
├── requirements.txt          # Dependencies
├── gan-model.ipynb          # Training notebook
├── hf_cache/                # HuggingFace cache
├── QUICKSTART.md            # This guide
└── README.md                # Documentation
```

---

## 🚀 Running

### Streamlit App (RECOMMENDED)
```bash
streamlit run app.py
```
**URL:** http://localhost:8501

### Command Line Script
```bash
python inference.py
```

---

## 🎯 Features

### 1️⃣ Generate Progression
- Progression stage-to-stage
- Download progression image

### 2️⃣ Generate Multiple Images  
- Pilih stage (0-3)
- Set jumlah (1-16)
- Smart selection: 2X → X most diverse
- **Download only** (no local storage)

### 3️⃣ Advanced Settings
**Variation Levels (EXTREME):**
- Normal: Alpha 3.0
- Tinggi: Alpha 5.0
- Sangat Tinggi: Alpha 7.0

---

## 🔥 How It Works

```
1. Generate 2X images dengan extreme traversal
2. Select X most diverse menggunakan greedy algorithm
3. Display & download langsung dari memory
```

---

## 💡 Key Points

✅ **EXTREME VARIATION** - Alpha 3.0-7.0 untuk maximum diversity  
✅ **Download Only** - Semua images hanya via download, tidak disimpan lokal  
✅ **No Warnings** - Semua deprecation warnings sudah fixed  
✅ **Smart Selection** - Greedy algorithm untuk pick most diverse images  
✅ **Fast** - CPU-friendly, works on any device  

---

## 📊 Model Info

- **Repository:** Arga23/dementia-cgan-mri (HuggingFace)
- **Architecture:** Conditional DCGAN
- **Image Size:** 128x128
- **Classes:** 4 dementia stages

---

## 🎉 Ready!

App running dengan EXTREME variation & download-only mode! 🚀
