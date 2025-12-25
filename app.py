"""
Dementia CGAN - Streamlit App
Generate synthetic MRI images for different dementia stages using Conditional GAN
Model dari HuggingFace Hub: Arga23/dementia-cgan-mri
"""

import streamlit as st
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io
from datetime import datetime
from pathlib import Path

# Import Generator architecture
try:
    from model_architecture import Generator
except ImportError:
    st.error("❌ model_architecture.py tidak ditemukan!")
    st.stop()

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Dementia CGAN Generator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E88E5;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
    }
    .stage-card {
        background-color: #F5F5F5;
        padding: 0.8rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
HF_REPO_ID = "Arga23/dementia-cgan-mri"

# ==================== LABEL SAFETY ====================
LABEL_MAPPING = {
    0: "Non-Dementia (Sehat)",
    1: "Very Mild Dementia (Sangat Ringan)",
    2: "Mild Dementia (Ringan)",
    3: "Moderate Dementia (Sedang)"
}

def safe_stage_name(stage_id):
    if stage_id not in LABEL_MAPPING:
        return "Unknown Stage"
    return LABEL_MAPPING[stage_id]

# Alias for backward compatibility in non-output logic
STAGE_NAMES = LABEL_MAPPING

STAGE_DESCRIPTIONS = {
    0: "Tidak ada tanda-tanda demensia, fungsi kognitif normal",
    1: "Gejala ringan, sedikit masalah memori",
    2: "Kesulitan mengingat dan mengorganisir pikiran",
    3: "Kesulitan signifikan dalam aktivitas sehari-hari"
}

REFERENCE_IMAGE_PATH = Path(__file__).resolve().parent / "real-dementia-mri.png"

# ==================== INITIALIZE MODEL ====================
@st.cache_resource
def load_model():
    """Load Generator model dari HuggingFace Hub"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load original model from HuggingFace Hub
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="cDCGAN_generator.pth",
            cache_dir="./hf_cache"
        )

        checkpoint = torch.load(model_path, map_location=device)
        G = Generator(
            z_dim=checkpoint['z_dim'],
            num_classes=checkpoint['num_classes'],
            img_channels=1
        ).to(device)
        G.load_state_dict(checkpoint['model'])
        G.eval()
        return G, checkpoint, device, None
    except Exception as e:
        return None, None, None, str(e)

# ==================== HELPER FUNCTIONS ====================
def diversity_score(img1, img2):
    """Calculate diversity between two images (L1 distance)"""
    return torch.mean(torch.abs(img1 - img2)).item()

def select_most_diverse(images_tensors, n_select):
    """Select n_select most diverse images from a larger set using greedy algorithm"""
    if len(images_tensors) <= n_select:
        return images_tensors
    
    # Start with first image
    selected = [images_tensors[0]]
    remaining = list(images_tensors[1:])
    
    # Greedy selection: pick images that maximize min distance to selected
    while len(selected) < n_select and remaining:
        max_min_dist = -1
        best_idx = 0
        
        for idx, img in enumerate(remaining):
            # Calculate minimum distance to all selected images
            min_dist = min(diversity_score(img, sel_img) for sel_img in selected)
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_idx = idx
        
        selected.append(remaining[best_idx])
        remaining.pop(best_idx)
    
    return selected

def generate_random(generator, z_dim, label, num_images, device):
    """
    🟢 FULLY RANDOM SAMPLING (Baseline Inference)
    Setiap gambar pakai latent vector BARU - tidak ada anchor/traversal.
    Paling robust dan variatif.
    """
    images_tensor = []
    with torch.no_grad():
        for _ in range(num_images):
            z = torch.randn(1, z_dim, 1, 1).to(device)
            img = generator(z, label)
            images_tensor.append(img)
    return images_tensor

def generate_images(generator, checkpoint, device, stage, num_images, variation_level="Tinggi"):
    """Convert tensor output to PIL images for display"""
    generator.eval()
    z_dim = checkpoint['z_dim']
    label = torch.tensor([stage], device=device)
    images = []
    descriptions = []

    # Use the fully random sampling function
    images_tensor = generate_random(generator, z_dim, label, num_images, device)
    
    for idx, img_tensor in enumerate(images_tensor, 1):
        img_array = img_tensor.squeeze().cpu().numpy()
        img_array = ((img_array + 1) / 2 * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_array, mode='L')
        images.append(img_pil)
        descriptions.append(f"Image {idx}")

    return images, descriptions

def create_image_grid(images, stage_name):
    """Create grid visualization dari generated images"""
    num_images = len(images)
    
    if num_images == 1:
        cols = 1
        rows = 1
    elif num_images <= 4:
        cols = 2
        rows = (num_images + 1) // 2
    else:
        cols = 4
        rows = (num_images + 3) // 4
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    
    # Handle different subplot configurations
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]
    
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        
        axes[row][col].imshow(img, cmap='gray')
        axes[row][col].axis('off')
        axes[row][col].set_title(f"Sample {idx + 1}", fontsize=12, fontweight='bold')
    
    # Hide empty subplots
    for idx in range(num_images, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row][col].axis('off')
    
    fig.suptitle(f"Generated Images - {stage_name}", 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    return fig

def rough_severity_score(img_pil):
    """
    Heuristic kasar:
    - stage tinggi → intensitas rata-rata lebih gelap (lebih banyak ruang kosong)
    """
    img = np.array(img_pil).astype(np.float32) / 255.0
    return float(img.mean())

def check_progression_consistency(scores):
    """
    Stage lebih tinggi seharusnya severity score lebih kecil
    """
    for i in range(len(scores) - 1):
        if scores[i + 1][1] > scores[i][1]:
            return False
    return True

def load_reference_severity():
    """Load reference MRI severity score for light sanity check."""
    if not REFERENCE_IMAGE_PATH.exists():
        return None, "Reference image not found"
    try:
        ref_img = Image.open(REFERENCE_IMAGE_PATH).convert("L")
        return rough_severity_score(ref_img), None
    except Exception as exc:  # pragma: no cover - defensive
        return None, str(exc)

def adjust_progression_labels(scores, start_stage, end_stage, reference_score=None, tolerance=0.02):
    """
    Re-map displayed stage labels based on severity ordering.
    Brighter images (lebih ringan) mendapat label stage yang lebih rendah.
    Jika reference_score tersedia, turunkan label jika score jauh lebih terang dari referensi.
    """
    desired_labels = list(range(start_stage, end_stage + 1))
    sorted_by_brightness = sorted(enumerate(scores), key=lambda x: -x[1][1])
    adjusted = [None] * len(scores)

    for (original_idx, (_, score)), desired_stage in zip(sorted_by_brightness, desired_labels):
        adjusted[original_idx] = desired_stage

    if reference_score is not None:
        for idx, (_, score_val) in enumerate(scores):
            if adjusted[idx] is None:
                continue
            if adjusted[idx] > start_stage and score_val > reference_score + tolerance:
                adjusted[idx] = max(start_stage, adjusted[idx] - 1)

    return adjusted

# ==================== MAIN APP ====================
def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Dementia CGAN Image Generator</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <b>🎯 Tentang Aplikasi Ini:</b><br>
    Generator gambar MRI sintetis untuk berbagai tahap demensia menggunakan <b>Conditional DCGAN</b>.
    Model diunduh langsung dari <b>HuggingFace Hub</b> (Arga23/dementia-cgan-mri).
    <br><br>
    <b>🔥 Smart Selection:</b> Generate 2X gambar menggunakan <b>Latent Traversal</b>, lalu pilih X gambar paling diverse untuk hasil maksimal!
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    with st.spinner("📦 Downloading model dari HuggingFace Hub..."):
        generator, checkpoint, device, error = load_model()
    
    if error:
        st.error(f"❌ Error loading model: {error}")
        st.info("💡 Pastikan koneksi internet aktif dan HuggingFace Hub dapat diakses")
        return
    
    st.markdown("""
    <div class="success-box">
    ✅ Model berhasil dimuat dari HuggingFace!<br>
    📊 Device: <b>{}</b> | Image Size: <b>{}x{}</b> | Classes: <b>{}</b>
    </div>
    """.format(
        device.upper(),
        checkpoint['img_size'],
        checkpoint['img_size'],
        checkpoint['num_classes']
    ), unsafe_allow_html=True)

    if 'reference_severity' not in st.session_state:
        ref_score, ref_error = load_reference_severity()
        st.session_state['reference_severity'] = ref_score
        st.session_state['reference_severity_error'] = ref_error
    
    # Sidebar - Stage Information
    with st.sidebar:
        st.header("ℹ️ Dementia Stages")
        st.write("Model dapat menghasilkan gambar MRI untuk 4 tahap demensia:")
        
        for stage_id in STAGE_NAMES.keys():
            st.markdown(f"""
            <div class="stage-card">
            <b>Stage {stage_id}:</b> {safe_stage_name(stage_id)}<br>
            <small>{STAGE_DESCRIPTIONS[stage_id]}</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.header("🔧 Model Info")
        st.write(f"**Repository:** {HF_REPO_ID}")
        st.write(f"**Architecture:** Conditional DCGAN")
        st.write(f"**Method:** Latent Traversal + Smart Selection")
        st.write(f"**z_dim:** {checkpoint['z_dim']}")
        st.write(f"**Image Size:** {checkpoint['img_size']}x{checkpoint['img_size']}")
    
    # ==================== SECTION 1: PROGRESSION (PRIORITY) ====================
    st.markdown("---")
    st.header("📈 Generate Progression")
    
    st.write("Generate urutan progression dari satu stage ke stage lainnya (fully random per stage)")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        start_stage = st.selectbox(
            "Start Stage:",
            options=list(STAGE_NAMES.keys()),
            format_func=lambda x: f"Stage {x}: {safe_stage_name(x)}",
            key="start_stage"
        )
    
    with col2:
        end_stage = st.selectbox(
            "End Stage:",
            options=list(STAGE_NAMES.keys()),
            format_func=lambda x: f"Stage {x}: {safe_stage_name(x)}",
            index=3,
            key="end_stage"
        )
    
    with col3:
        if st.button("📊 Generate Progression", width="stretch", type="primary"):
            if end_stage < start_stage:
                st.warning("⚠️ End stage harus >= start stage")
            else:
                with st.spinner("Generating progression with Latent Traversal + Diversity..."):
                    progression_images = []
                    
                    for stage in range(start_stage, end_stage + 1):
                        # Fully random sampling: one random image per stage
                        imgs, _ = generate_images(
                            generator, checkpoint, device, stage, 1
                        )
                        progression_images.append((stage, imgs[0]))
                    scores = []
                    for stage, img in progression_images:
                        scores.append((stage, rough_severity_score(img)))

                    reference_score = st.session_state.get('reference_severity')
                    adjusted_labels = adjust_progression_labels(
                        scores,
                        start_stage,
                        end_stage,
                        reference_score=reference_score
                    )

                    progression_state = []
                    for idx, (stage, img) in enumerate(progression_images):
                        display_stage = adjusted_labels[idx] if adjusted_labels[idx] is not None else stage
                        progression_state.append({
                            "original_stage": stage,
                            "display_stage": display_stage,
                            "image": img,
                            "score": scores[idx][1]
                        })

                    consistency_ok = check_progression_consistency(scores)
                    labels_corrected = any(item['display_stage'] != item['original_stage'] for item in progression_state)

                    st.session_state['progression_state'] = progression_state
                    st.session_state['progression_consistent'] = consistency_ok
                    st.session_state['progression_labels_corrected'] = labels_corrected
                    st.session_state['progression_scores'] = scores
                    st.session_state['progression_range'] = (start_stage, end_stage)

                    if not consistency_ok or labels_corrected:
                        st.info("✅ Progression generated dengan auto label check (sanity-corrected).")
                    else:
                        st.success("✅ Progression generated dengan Latent Traversal!")
    
    # Display progression results
    if 'progression_state' in st.session_state:
        progression_state = st.session_state['progression_state']
        consistency_ok = st.session_state.get('progression_consistent', True)
        labels_corrected = st.session_state.get('progression_labels_corrected', False)
        scores = st.session_state.get('progression_scores', [])
        ref_score = st.session_state.get('reference_severity')
        ref_error = st.session_state.get('reference_severity_error')
        start_saved, end_saved = st.session_state.get('progression_range', (start_stage, end_stage))

        if ref_error:
            st.warning(f"⚠️ Reference check skipped: {ref_error}")
        elif ref_score is not None:
            st.caption(f"👀 Reference severity (real-dementia-mri.png): {ref_score:.4f}")

        if not consistency_ok or labels_corrected:
            st.warning("Label auto-adjusted berdasarkan severity order + reference sanity check.")
        else:
            st.success("Progression severity sudah konsisten (lebih gelap = lebih severe).")

        if scores:
            score_lines = []
            for item in progression_state:
                relabel_note = ""
                if item['display_stage'] != item['original_stage']:
                    relabel_note = f" • relabeled from Stage {item['original_stage']}"
                score_lines.append(
                    f"Stage {item['display_stage']} ({safe_stage_name(item['display_stage'])}) → {item['score']:.4f}{relabel_note}"
                )
            st.caption("\n".join(score_lines))

        cols = st.columns(len(progression_state))
        for idx, item in enumerate(progression_state):
            stage_label = safe_stage_name(item['display_stage'])
            caption = f"Stage {item['display_stage']}\n{stage_label}"
            if item['display_stage'] != item['original_stage']:
                caption += f"\n(auto from Stage {item['original_stage']})"
            cols[idx].image(item['image'], caption=caption, width='stretch')
        
        # Create combined image for download
        fig, axes = plt.subplots(1, len(progression_state), figsize=(len(progression_state) * 3, 3))
        if len(progression_state) == 1:
            axes = [axes]
        
        for idx, item in enumerate(progression_state):
            stage_to_show = item['display_stage']
            axes[idx].imshow(item['image'], cmap='gray')
            axes[idx].axis('off')
            axes[idx].set_title(
                f"Stage {stage_to_show}\n{safe_stage_name(stage_to_show)}",
                fontsize=10,
                fontweight='bold'
            )
        
        fig.suptitle("Dementia Progression", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        
        st.download_button(
            label="📥 Download Progression Image",
            data=buf,
            file_name=f"dementia_progression_stage_{start_saved}_to_{end_saved}.png",
            mime="image/png",
            width="stretch"
        )
    
    # ==================== SECTION 2: GENERATE IMAGES ====================
    st.markdown("---")
    st.header("🎨 Generate Multiple Images")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Settings")
        
        # Stage selection
        selected_stage = st.selectbox(
            "Pilih Tahap Demensia:",
            options=list(STAGE_NAMES.keys()),
            format_func=lambda x: f"Stage {x}: {safe_stage_name(x)}",
            help="Pilih tahap demensia untuk menghasilkan gambar MRI sintetis"
        )
        
        # Show selected stage description
        st.info(f"📝 {STAGE_DESCRIPTIONS[selected_stage]}")
        
        # Number of images
        num_images = st.slider(
            "Jumlah Gambar:",
            min_value=1,
            max_value=16,
            value=4,
            help="Jumlah gambar yang akan di-generate dan dipilih yang paling diverse"
        )
        
        # Advanced settings expander
        with st.expander("⚙️ Advanced Settings", expanded=True):
            st.markdown("**🎯 Generation Method:**")
            st.success("🟢 Fully Random Sampling (baseline inference)")
            st.caption("Setiap gambar pakai latent baru. Tidak ada traversal/anchor.")

            st.markdown("---")
            st.markdown("**🎲 Variasi:**")
            st.caption("Tidak digunakan di random sampling (semua latent baru)")

            st.markdown("---")
            st.markdown(f"**📈 Generation Process:**")
            st.info(f"1️⃣ Generate **{num_images}** images dengan latent acak\n\n2️⃣ Tidak ada seleksi/diversity filter\n\n3️⃣ Download langsung (no storage)")
        
        st.write(f"Akan generate **{num_images} gambar** (random sampling)")
        st.caption(f"🎯 Stage: {safe_stage_name(selected_stage)}")
        
        # Generate button
        if st.button("🚀 Generate Images", type="primary", width="stretch"):
            with st.spinner(f"⏳ Generating {num_images} images (fully random)..."):
                generated_images, saved_paths = generate_images(
                    generator, 
                    checkpoint, 
                    device, 
                    selected_stage, 
                    num_images,
                    variation_level=None
                )
                
                st.session_state['generated_images'] = generated_images
                st.session_state['saved_paths'] = saved_paths
                st.session_state['current_stage'] = selected_stage
                st.session_state['current_stage_name'] = safe_stage_name(selected_stage)
                
                st.success(f"✅ Berhasil generate {num_images} gambar (fully random)!")
    
    with col2:
        st.subheader("📊 Generated Results")
        
        if 'generated_images' in st.session_state:
            images = st.session_state['generated_images']
            stage_name = st.session_state['current_stage_name']
            
            # Create and display grid
            fig = create_image_grid(images, stage_name)
            st.pyplot(fig)
            
            # Download button
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            
            st.download_button(
                label="📥 Download All Images (Grid)",
                data=buf,
                file_name=f"dementia_stage_{st.session_state['current_stage']}_generated.png",
                mime="image/png",
                width="stretch"
            )
            
            # Individual images
            st.markdown("---")
            st.subheader("📷 Individual Images")
            
            cols_per_row = 4
            for i in range(0, len(images), cols_per_row):
                cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    idx = i + j
                    if idx < len(images):
                        img_pil = images[idx]
                        
                        # Convert to bytes for download
                        buf = io.BytesIO()
                        img_pil.save(buf, format='PNG')
                        buf.seek(0)
                        
                        with cols[j]:
                            st.image(img_pil, caption=f"Sample {idx+1}", width='stretch')
                            st.download_button(
                                label="⬇️ Download",
                                data=buf,
                                file_name=f"stage_{st.session_state['current_stage']}_sample_{idx+1}.png",
                                mime="image/png",
                                key=f"download_{idx}",
                                width="stretch"
                            )
        else:
            st.info("👈 Pilih stage dan jumlah gambar, lalu klik 'Generate Images' untuk mulai")
            st.image("https://via.placeholder.com/600x400/E3F2FD/1E88E5?text=Preview+Area", width='stretch')
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
    <p>🔬 Model: <b>Conditional DCGAN</b> trained on Alzheimer's MRI Dataset</p>
    <p>🤗 HuggingFace: <b>Arga23/dementia-cgan-mri</b></p>
    <p>🔥 Method: <b>Latent Traversal (Extreme) + Smart Diversity Selection</b></p>
    <p>📥 Download: All images available for download (no local storage)</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== RUN APP ====================
if __name__ == "__main__":
    main()
