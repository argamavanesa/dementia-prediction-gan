# 🏥 Dementia Brain MRI Generation using Conditional DCGAN

## 🎯 Overview
This project uses **Conditional Deep Convolutional GAN (DCGAN)** to generate synthetic brain MRI images for different Alzheimer's dementia severity stages.

## 🧠 Key Concept

**Conditional GAN approach:**
```
noise + stage_label → synthetic_brain_MRI
```

This is useful for medical imaging because:
- ✅ Data augmentation for rare classes
- ✅ Balanced dataset generation
- ✅ Research and educational purposes
- ✅ No paired data required

## 📊 Dementia Stages (OASIS Dataset)

- **Stage 0**: Non-Dementia (Normal)
- **Stage 1**: Very Mild Dementia
- **Stage 2**: Mild Dementia
- **Stage 3**: Moderate Dementia

## 🏗️ Architecture

### Generator: Conditional DCGAN
- **Input:** Noise vector (100-dim) + Label embedding (4-dim)
- **Output:** 128x128 grayscale MRI image
- **Features:** ConvTranspose2d + BatchNorm + ReLU
- **Activation:** Tanh (output range [-1, 1])

### Discriminator: Conditional DCGAN
- **Input:** Image (128x128) + Spatial label embedding
- **Output:** Real/Fake logits (single value)
- **Features:** Conv2d + LeakyReLU (no BatchNorm)
- **Loss:** BCEWithLogitsLoss

## 🔧 Training Configuration

- **Epochs:** 10 (configurable)
- **Batch Size:** 16
- **Learning Rate:** G=2e-4, D=5e-5
- **Optimizer:** Adam (β1=0.5, β2=0.999)
- **Image Size:** 128x128
- **Latent Dimension:** 100
- **Mixed Precision:** AMP enabled (GPU only)
- **Sampling:** Balanced (WeightedRandomSampler)

## 🚀 Usage

### Training
```bash
# Open and run the notebook: gan-model.ipynb
# All cells are sequential and self-contained
# Runs on Kaggle with OASIS dataset
```

### Inference (Local - downloads from HuggingFace)
```bash
# Install dependencies
pip install torch torchvision huggingface_hub pillow

# Download model_architecture.py and inference.py from HuggingFace
# Or run Step 5 in notebook to generate them

# Run inference
python inference.py
```

The script will:
1. Download the model from HuggingFace Hub automatically
2. Generate samples for all 4 dementia stages
3. Save as `generated_stage_0.png`, `generated_stage_1.png`, etc.

### Inference (Python code)
```python
import torch
from huggingface_hub import hf_hub_download
from model_architecture import Generator
from PIL import Image

# Download model from HuggingFace
model_path = hf_hub_download(
    repo_id="Arga23/dementia-cgan-mri",
    filename="cDCGAN_generator.pth"
)

# Load checkpoint
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = torch.load(model_path, map_location=device)

# Initialize Generator
G = Generator(
    z_dim=checkpoint['z_dim'],
    num_classes=checkpoint['num_classes'],
    img_channels=1
).to(device)
G.load_state_dict(checkpoint['model'])
G.eval()

# Generate image for specific stage
with torch.no_grad():
    z = torch.randn(1, 100, 1, 1).to(device)
    label = torch.tensor([2]).to(device)  # Stage 2 (Mild Dementia)
    img = G(z, label)
    img = (img.squeeze().cpu() + 1) / 2
    img = torch.clamp(img, 0, 1)
    Image.fromarray((img.numpy() * 255).astype('uint8'), mode='L').save('generated_mild.png')
```

## 📁 Project Structure

```
dementia-prediction-gan/
├── gan-model.ipynb          # Main training notebook (Conditional DCGAN)
├── inference.py             # Inference script (downloads from HuggingFace)
├── model_architecture.py    # Generator architecture
├── requirements.txt         # Dependencies
├── model_output/            # Generated after training
│   ├── cDCGAN_generator.pth
│   ├── model_architecture.py
│   ├── inference.py
│   └── training_history.csv
└── README.md
```

## 📦 Requirements

```
torch>=2.0.0
torchvision
huggingface_hub
pillow
matplotlib
numpy
pandas
scipy
```

Install:
```bash
pip install -r requirements.txt
```

## 🔬 Scientific Justification

### Why Conditional DCGAN for Medical Imaging?

1. **Controlled Generation**
   - Generate specific dementia stages on demand
   - Label conditioning ensures stage-specific features

2. **Data Augmentation**
   - Address class imbalance in medical datasets
   - Generate synthetic samples for rare classes

3. **No Paired Data Required**
   - Only needs labeled images (stage 0, 1, 2, 3)
   - No need for paired before/after images

4. **Proven Architecture**
   - DCGAN is stable and well-established
   - ConvTranspose layers generate high-quality images
   - BatchNorm improves training stability

## 🎯 HuggingFace Model

Model available at: [Arga23/dementia-cgan-mri](https://huggingface.co/Arga23/dementia-cgan-mri)

Files included:
- `cDCGAN_generator.pth` - Generator weights
- `model_architecture.py` - PyTorch architecture code
- `inference.py` - Ready-to-use inference script
- `README.md` - Documentation

## ⚠️ Disclaimer

**FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY**

Generated images should **NOT** be used for:
- Clinical diagnosis
- Medical decision making
- Patient treatment plans
- Any clinical application

This model is trained on the OASIS dataset for research purposes only.

### Evaluation Metrics
- **L1 Loss:** Structural similarity
- **Adversarial Loss:** Realism
- **FID Score:** Image quality (optional)

## 🎓 Academic References

- **Pix2Pix:** [Image-to-Image Translation with Conditional GANs (Isola et al., 2017)](https://arxiv.org/abs/1611.07004)
- **U-Net:** [Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015)](https://arxiv.org/abs/1505.04597)
- **OASIS Dataset:** [Open Access Series of Imaging Studies](https://www.oasis-brains.org/)

## ⚠️ Disclaimer

This project is for **research and educational purposes only**. Generated predictions should NOT be used for clinical diagnosis or medical decisions. Always consult qualified healthcare professionals.

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📧 Contact

For questions or collaborations, please open an issue on GitHub.

---

**Status:** ✅ Training pipeline complete | 🚧 Deployment in progress
