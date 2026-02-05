import streamlit as st
import fal_client
import io
import zipfile
import os
import asyncio
import base64
import random
import re
import requests
from PIL import Image

# --- Configuration ---
st.set_page_config(page_title="Jewelry AI Automator", layout="wide")

# Endpoints
VISION_ENDPOINT = "fal-ai/llava-v1.5-13b"
FLUX_INPAINT_ENDPOINT = "fal-ai/flux-pro/v1.1/inpainting" # Level 2: Component Precision
FAST_SAM_ENDPOINT = "fal-ai/fast-sam" # For auto-segmentation

def get_image_base64(image):
    """Convert PIL Image to base64 for API upload."""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode('utf-8')

# --- Main App ---
st.title("💎 JewelBench Level 2: Component Precision")
st.markdown("""
**New Workflow (Phase 2):**
1.  **Vision ID:** Identifies the jewelry.
2.  **Auto-Segmentation:** Isolates specific parts (e.g., "Shank", "Diamond").
3.  **Component Inpainting:** Swaps ONLY the selected part.
""")

# Sidebar
with st.sidebar:
    api_key_input = st.text_input("Fal.ai API Key", type="password")
    if api_key_input:
        os.environ["FAL_KEY"] = api_key_input
    elif "FAL_KEY" in st.secrets:
        os.environ["FAL_KEY"] = st.secrets["FAL_KEY"]
    
    st.markdown("### Batch Settings")
    batch_size = st.select_slider("Variations per Component", options=[1, 5, 10], value=5)
    
    st.markdown("### Target Component")
    target_part = st.selectbox("What to change?", ["Shank/Band", "Center Stone", "Setting/Head"])
    
    st.markdown("### New Style")
    new_style = st.text_input("Describe new style:", "Twisted Pavé Gold")

uploaded_file = st.file_uploader("Upload Master Jewelry Photo", type=['png', 'jpg', 'jpeg', 'webp'])

if uploaded_file:
    # Display Original
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(uploaded_file, caption="Master Design", use_container_width=True)
        input_img = Image.open(uploaded_file)
        img_b64 = get_image_base64(input_img)

    with col2:
        if st.button("🚀 Segment & Swap", type="primary"):
            if not os.environ.get("FAL_KEY"):
                st.error("Please provide Fal.ai API Key.")
                st.stop()

            # 1. AUTO-SEGMENTATION (The "Anatomy" Step)
            with st.spinner(f"🔍 Segmenting {target_part}..."):
                try:
                    # We use Fast-SAM with a text prompt to find the part
                    sam_prompt = target_part.lower().replace("/", " or ")
                    sam_handler = fal_client.submit(
                        FAST_SAM_ENDPOINT,
                        arguments={
                            "image_url": img_b64,
                            "text_prompt": sam_prompt
                        }
                    )
                    sam_result = sam_handler.get()
                    
                    # Fast-SAM returns multiple masks, we usually take the first/best one
                    # or the combined one. For simplicity in this demo, we assume index 0 is valid.
                    if sam_result['masks']:
                        mask_url = sam_result['masks'][0]['url']
                        st.image(mask_url, caption=f"Detected {target_part}", width=150)
                    else:
                        st.error("Could not detect component. Try a clearer image.")
                        st.stop()
                        
                except Exception as e:
                    st.error(f"Segmentation Failed: {e}")
                    st.stop()

            # 2. GENERATE COMPONENT VARIATIONS (The "Flux" Step)
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            async def run_generations():
                tasks = []
                # We prompt Flux to fill the MASKED area with the NEW STYLE
                # The prompt must describe the WHOLE image context + the specific change
                full_prompt = f"A professional jewelry photo. The {target_part} is {new_style}. High quality, photorealistic, 8k, seamless blend."
                
                for _ in range(batch_size):
                    tasks.append(fal_client.submit_async(
                        FLUX_INPAINT_ENDPOINT,
                        arguments={
                            "prompt": full_prompt,
                            "image_url": img_b64,
                            "mask_url": mask_url,
                            "strength": 1.0, # 1.0 = completely replace the masked area
                            "guidance_scale": 7.5
                        }
                    ))
                
                # Run in chunks
                chunk_size = 4
                completed = 0
                
                for i in range(0, len(tasks), chunk_size):
                    chunk = tasks[i:i+chunk_size]
                    batch_responses = await asyncio.gather(*chunk)
                    
                    for resp_handler in batch_responses:
                        try:
                            res = await resp_handler.get()
                            img_url = res['images'][0]['url']
                            img_data = requests.get(img_url).content
                            results.append(img_data)
                        except Exception as e:
                            print(f"Gen Error: {e}")
                    
                    completed += len(chunk)
                    progress_bar.progress(min(completed / batch_size, 1.0))
                    status_text.text(f"Generated {len(results)} / {batch_size} designs...")

            asyncio.run(run_generations())

            # 3. DISPLAY & DOWNLOAD
            if results:
                st.success("Component Swap Complete!")
                
                # Gallery
                cols = st.columns(4)
                for idx, img_bytes in enumerate(results):
                    with cols[idx % 4]:
                        st.image(img_bytes, caption=f"Var #{idx+1}", use_container_width=True)
                
                # Zip
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    for idx, img_bytes in enumerate(results):
                        zf.writestr(f"swap_{idx+1}.jpg", img_bytes)
                
                st.download_button(
                    "Download All Variations",
                    data=zip_buffer.getvalue(),
                    file_name="jewelbench_component_swaps.zip",
                    mime="application/zip"
                )
