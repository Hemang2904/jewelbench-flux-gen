import streamlit as st
import requests
import io
import zipfile
import hashlib
import time
from PIL import Image
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration ---
st.set_page_config(page_title="JewelBench - Flux 2.0 Generator", layout="wide")

# Constants for Flux 2.0 (Placeholder URL - replace with actual endpoint)
# Note: As of late 2024/early 2025, "Flux 2.0" API specifics can vary by provider (e.g., Replicate, Fal.ai, etc.)
# This implementation assumes a generic POST endpoint structure common to modern diffusion APIs.
# You might need to adjust the payload structure based on the specific provider you are using.
API_URL_TEMPLATE = "https://api.bfl.ml/v1/flux-pro-1.1-ultra" # Example endpoint, user must confirm specific provider

def get_image_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def calculate_hash(image_bytes):
    return hashlib.md5(image_bytes).hexdigest()

def generate_image(api_key, prompt, image_input=None, aspect_ratio="3:4"):
    """
    Calls the Flux 2.0 API.
    Adjust headers/payload based on the specific provider documentation.
    """
    headers = {
        "Content-Type": "application/json",
        "X-Key": api_key # specific header depends on provider
    }
    
    # Enforcing 3/4 angle and jewelry specific keywords
    enhanced_prompt = f"{prompt}, professional jewelry photography, 3/4 view, isometric angle, high detail, 8k, neutral studio background"
    
    payload = {
        "prompt": enhanced_prompt,
        "aspect_ratio": aspect_ratio,
        "safety_tolerance": 2,
        "output_format": "jpeg"
    }
    
    # If image input exists (img2img), add it to payload
    if image_input:
        # Note: API implementation for img2img varies greatly. 
        # Some accept URL, some base64. Assuming base64 for this template.
        payload["image_prompt"] = get_image_base64(image_input)
        payload["image_strength"] = 0.65 # Keep strong coherence to original shape/angle

    try:
        # This is a synchronous blocking call example. 
        # Real production apps might use async polling for Flux/Pro models.
        response = requests.post(API_URL_TEMPLATE, json=payload, headers=headers)
        response.raise_for_status()
        
        # Assume response returns a URL or base64. 
        # Adjust parsing logic based on actual API response structure.
        # Example: {"result": {"sample": "https://..."}}
        result_url = response.json().get("result", {}).get("sample")
        
        if result_url:
            img_data = requests.get(result_url).content
            return img_data
        else:
            return None
            
    except Exception as e:
        st.error(f"Generation failed: {str(e)}")
        return None

# --- UI Layout ---
st.title("💎 JewelBench: Flux 2.0 Bulk Generator")
st.markdown("Generate bulk jewelry variations with consistent 3/4 angles using Flux 2.0.")

with st.sidebar:
    api_key = st.text_input("Flux API Key", type="password")
    st.info("Ensure you have sufficient credits.")
    
    batch_size = st.select_slider("Batch Size", options=[25, 50, 75, 100], value=25)
    
    st.markdown("### Settings")
    guidance = st.slider("Creativity (Guidance)", 1.0, 10.0, 3.5)

# --- Main Inputs ---
col1, col2 = st.columns([1, 1])

with col1:
    input_mode = st.radio("Input Mode", ["Text Only", "Image Only", "Text + Image"])
    
    text_prompt = ""
    if "Text" in input_mode:
        text_prompt = st.text_area("Description", "Gold diamond ring, art deco style, intricate bezel setting")

    uploaded_file = None
    if "Image" in input_mode:
        uploaded_file = st.file_uploader("Reference Image (determines angle/structure)", type=["jpg", "png"])
        if uploaded_file:
            st.image(uploaded_file, width=200)

# --- Generation Logic ---
if st.button("Generate Batch", type="primary"):
    if not api_key:
        st.error("Please enter an API Key.")
        st.stop()
        
    if "Image" in input_mode and not uploaded_file:
        st.error("Please upload a reference image.")
        st.stop()

    input_img = Image.open(uploaded_file) if uploaded_file else None
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    generated_images = []
    seen_hashes = set()
    
    # Using ThreadPool for parallel requests (be mindful of API rate limits)
    # Flux API often allows concurrent requests.
    
    status_text.text("Starting generation process...")
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for i in range(batch_size):
            # Vary seed or slight prompt noise could be added here if API supports it
            # to ensure variations. For now, rely on model stochasticity.
            futures.append(executor.submit(generate_image, api_key, text_prompt, input_img))
            
        completed = 0
        for future in as_completed(futures):
            img_bytes = future.result()
            if img_bytes:
                img_hash = calculate_hash(img_bytes)
                if img_hash not in seen_hashes:
                    seen_hashes.add(img_hash)
                    generated_images.append(img_bytes)
                else:
                    st.warning("Duplicate image detected and skipped.")
            
            completed += 1
            progress_bar.progress(completed / batch_size)
            status_text.text(f"Generated {len(generated_images)} / {batch_size} unique variations")

    # --- Results & Download ---
    if generated_images:
        st.success(f"Batch complete! {len(generated_images)} unique images created.")
        
        # ZIP Creation
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            for idx, img_data in enumerate(generated_images):
                zip_file.writestr(f"variation_{idx+1}.jpg", img_data)
        
        st.download_button(
            label="Download All Images (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="jewelbench_variations.zip",
            mime="application/zip"
        )
        
        # Gallery Preview (first 8)
        st.markdown("### Preview (First 8)")
        cols = st.columns(4)
        for i, img_data in enumerate(generated_images[:8]):
            with cols[i % 4]:
                st.image(img_data, use_container_width=True)
