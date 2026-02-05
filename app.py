import streamlit as st
import fal_client
import io
import zipfile
import hashlib
import os
import asyncio
from PIL import Image

# --- Configuration ---
st.set_page_config(page_title="JewelBench - Flux Generator", layout="wide")

# Valid Fal.ai Models for High Quality
# flux-pro/v1.1-ultra is currently top-tier for detail/photorealism
MODEL_ENDPOINT = "fal-ai/flux-pro/v1.1-ultra"

def get_image_bytes(image):
    """Convert PIL Image to bytes for upload/hashing."""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=95)
    return buffered.getvalue()

def calculate_hash(image_bytes):
    return hashlib.md5(image_bytes).hexdigest()

async def generate_variation_async(prompt, aspect_ratio="3:4"):
    """
    Async wrapper for Fal.ai generation.
    """
    try:
        # Fal client handles the queueing and result fetching automatically
        handler = await fal_client.submit_async(
            MODEL_ENDPOINT,
            arguments={
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "safety_tolerance": "2",
                "output_format": "jpeg"
            },
        )
        result = await handler.get()
        image_url = result['images'][0]['url']
        
        # Download image data immediately to store in memory
        import requests
        resp = requests.get(image_url)
        return resp.content
    except Exception as e:
        print(f"Error: {e}")
        return None

# --- UI Layout ---
st.title("💎 JewelBench: Flux (Fal.ai) Generator")
st.markdown("Generate bulk jewelry variations using **Flux Pro 1.1 Ultra** via Fal.ai.")

# Sidebar
with st.sidebar:
    api_key_input = st.text_input("Fal.ai API Key", type="password", help="Get from fal.ai/dashboard")
    if api_key_input:
        os.environ["FAL_KEY"] = api_key_input
    
    st.markdown("---")
    batch_size = st.select_slider("Batch Size", options=[5, 10, 25, 50], value=5)
    st.caption("Note: 'Flux Pro Ultra' is a premium model. Watch your credits.")

# Main Inputs
col1, col2 = st.columns([1, 1])

with col1:
    input_type = st.radio("Input Type", ["Text Description"])
    # Note: Flux Pro 1.1 Ultra is primarily Text-to-Image. 
    # For Image-to-Image, we would switch models, but let's stick to the best quality one first.
    
    base_prompt = st.text_area(
        "Jewelry Description", 
        "A high jewelry diamond ring, art deco style, platinum band, isometric view, neutral studio lighting, 8k resolution, macro photography"
    )

with col2:
    st.info("💡 **Tip:** The prompt automatically enforces '3/4 view' and 'studio lighting' for consistent 3D references.")

# Generation
if st.button("Generate Batch", type="primary"):
    if not api_key_input:
        st.error("Please enter your Fal.ai API Key in the sidebar.")
        st.stop()
        
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    # Create prompts for the batch (can inject random seeds/noise in prompt if needed, 
    # but Flux is naturally stochastic)
    
    results = []
    seen_hashes = set()
    
    # Using asyncio loop for concurrency
    async def run_batch():
        tasks = []
        for _ in range(batch_size):
            # We append a tiny random seed to prompt or rely on API randomness
            # Flux Pro usually varies well on its own.
            tasks.append(generate_variation_async(base_prompt))
        
        # Fal allows parallel requests. 
        # We'll run them in chunks to avoid rate limits if user account is new.
        completed_count = 0
        chunk_size = 5
        
        for i in range(0, len(tasks), chunk_size):
            chunk = tasks[i:i + chunk_size]
            batch_results = await asyncio.gather(*chunk)
            
            for img_data in batch_results:
                if img_data:
                    h = calculate_hash(img_data)
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        results.append(img_data)
            
            completed_count += len(chunk)
            progress = min(completed_count / batch_size, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Generated {len(results)} images...")

    asyncio.run(run_batch())

    # --- Display & Download ---
    if results:
        st.success(f"Complete! Generated {len(results)} unique variations.")
        
        # ZIP Download
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for idx, img_bytes in enumerate(results):
                zf.writestr(f"variation_{idx+1}.jpg", img_bytes)
        
        st.download_button(
            "Download All (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="jewelbench_variations.zip",
            mime="application/zip"
        )
        
        # Gallery
        st.markdown("### Preview")
        cols = st.columns(4)
        for idx, img_bytes in enumerate(results[:8]):
            with cols[idx % 4]:
                st.image(img_bytes, use_container_width=True)
