# code use gpu

import os
import zipfile
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import numpy as np
import streamlit as st
import requests
import shutil
from tqdm import tqdm

# -------------------------------
# SECTION 1 – SETUP
# -------------------------------
st.set_page_config(page_title="Priyanka - Human Face Generator", layout="wide")
st.title("🧠 Priyanka - Human Face Generator (AI + DL Project)")
st.markdown("Generate realistic human faces using **StyleGAN2-ADA** trained on Celebrity Faces Dataset (and your own).")

ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "models")
DATA_DIR = os.path.join(ROOT_DIR, "data")
GENERATED_DIR = os.path.join(ROOT_DIR, "generated_faces")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(GENERATED_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
st.info(f"Running on: {device.upper()}")

# -------------------------------
# SECTION 2 – UPLOAD DATASET
# -------------------------------
st.header("📦 Step 1: Upload Celebrity Faces Dataset (.zip)")
uploaded_file = st.file_uploader("Upload dataset zip (max 200MB)", type=["zip"])

if uploaded_file:
    zip_path = os.path.join(DATA_DIR, "celebrity_faces.zip")
    with open(zip_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.join(DATA_DIR, "celebrity_faces_raw"))
    st.success("✅ Dataset extracted successfully!")

# -------------------------------
# SECTION 3 – UPLOAD USER IMAGES
# -------------------------------
st.header("📸 Step 2: Add Your Own Face Images")
st.markdown("Upload square images, 512×512 px recommended.")

user_imgs = st.file_uploader("Upload your face images", type=["jpg","png","jpeg"], accept_multiple_files=True)

if user_imgs:
    user_dir = os.path.join(DATA_DIR, "celebrity_faces_raw", "user_faces")
    os.makedirs(user_dir, exist_ok=True)
    for img in user_imgs:
        img_path = os.path.join(user_dir, img.name)
        with open(img_path, "wb") as f:
            f.write(img.getbuffer())
    st.success(f"✅ {len(user_imgs)} personal images added.")

# -------------------------------
# SECTION 4 – PREPROCESS IMAGES
# -------------------------------
st.header("🧹 Step 3: Preprocess Images")

def preprocess_images(input_folder, output_folder, img_size=512):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(img_size)
    ])
    os.makedirs(output_folder, exist_ok=True)
    for img_name in tqdm(os.listdir(input_folder)):
        try:
            img_path = os.path.join(input_folder, img_name)
            img = Image.open(img_path).convert("RGB")
            img = transform(img)
            img.save(os.path.join(output_folder, img_name))
        except:
            continue

if st.button("Preprocess Dataset"):
    raw_path = os.path.join(DATA_DIR, "celebrity_faces_raw")
    processed_path = os.path.join(DATA_DIR, "celebrity_faces_processed")
    preprocess_images(raw_path, processed_path)
    st.success("✅ Image preprocessing complete!")

# -------------------------------
# SECTION 5 – LOAD PRETRAINED MODEL
# -------------------------------
st.header("🧠 Step 4: Load Pretrained FFHQ Model")

MODEL_PATH = os.path.join(MODEL_DIR, "ffhq.pkl")
if not os.path.exists(MODEL_PATH):
    st.warning("Downloading pretrained model (~380 MB)...")
    url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
    r = requests.get(url, stream=True)
    with open(MODEL_PATH, "wb") as f:
        shutil.copyfileobj(r.raw, f)
    st.success("✅ Pretrained model downloaded!")

st.info("Pretrained FFHQ model loaded! Ready for face generation.")

# -------------------------------
# SECTION 6 – GENERATE FACES
# -------------------------------
st.header("🎨 Step 5: Generate Faces")

seed = st.slider("Random seed", 0, 1000, 42)
num_faces = st.number_input("Number of faces to generate", 1, 10, 3)

if st.button("Generate Faces"):
    import legacy
    import dnnlib
    from torch_utils import gen_utils

    device_t = torch.device(device)

    # Load pretrained generator
    with open(MODEL_PATH, 'rb') as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device_t)  # type: ignore

    torch.manual_seed(seed)
    for i in range(num_faces):
        z = torch.randn(1, G.z_dim, device=device_t)
        img = G(z, None, truncation_psi=0.5, noise_mode='const')
        img = (img.clamp(-1,1) + 1) * 0.5  # scale to [0,1]
        save_path = os.path.join(GENERATED_DIR, f"face_{i}.png")
        save_image(img, save_path)
        st.image(save_path, caption=f"Generated Face {i+1}")

    st.success("✅ Face generation complete!")

# -------------------------------
# SECTION 7 – OPTIONAL: TRAIN / FINE-TUNE
# -------------------------------
st.header("🧩 Step 6: Fine-Tune Model (Optional)")
st.markdown("""
You can fine-tune StyleGAN2-ADA on your dataset using NVIDIA's official training script:

```bash
python train.py --outdir=training-runs --data=data/celebrity_faces_processed --gpus=1 --cfg=stylegan2 --mirror=1 --resume=models/ffhq.pkl
""")