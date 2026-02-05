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
FLUX_EDIT_ENDPOINT = "fal-ai/flux-2-pro/edit"

def get_image_base64(image):
    """Convert PIL Image to base64 for API upload."""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode('utf-8')

def parse_dynamic_prompt(prompt):
    """
    Parses prompts with syntax like [option1 | option2 | option3].
    Returns a single resolved prompt by picking one option at random.
    """
    while '[' in prompt and ']' in prompt:
        match = re.search(r'\[(.*?)\]', prompt)
        if match:
            options = match.group(1).split('|')
            choice = random.choice(options).strip()
            prompt = prompt.replace(match.group(0), choice, 1)
    return prompt

# --- Main App ---
st.title("💎 Jewelry AI Automator: Identify & Redesign")
st.markdown("""
**Workflow:**
1.  **Vision ID:** AI identifies the metal, motif, and key features of your master photo.
2.  **Flux Redesign:** Automatically redesigns the piece into new styles (Art Deco, Minimalist, etc.) using `fal-ai/flux-2-pro/edit`.
""")

# Sidebar
with st.sidebar:
    api_key_input = st.text_input("Fal.ai API Key", type="password")
    if api_key_input:
        os.environ["FAL_KEY"] = api_key_input
    elif "FAL_KEY" in st.secrets:
        os.environ["FAL_KEY"] = st.secrets["FAL_KEY"]
    
    st.markdown("### Batch Settings")
    batch_size = st.select_slider("Variations per Style", options=[1, 5, 10, 25], value=5)
    
    st.markdown("### Redesign Styles")
    style_art_deco = st.checkbox("Art Deco (Gold #FFD700)", value=True)
    style_minimal = st.checkbox("Minimalist (Rose #E6C2B4)", value=False)
    style_pavé = st.checkbox("Pavé Setting (Platinum #E5E4E2)", value=False)

uploaded_file = st.file_uploader("Upload Master Jewelry Photo", type=['png', 'jpg', 'jpeg', 'webp'])

if uploaded_file:
    # Display Original
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(uploaded_file, caption="Master Design", use_container_width=True)
        input_img = Image.open(uploaded_file)
        img_b64 = get_image_base64(input_img)

    with col2:
        if st.button("🚀 Analyze & Generate Batch", type="primary"):
            if not os.environ.get("FAL_KEY"):
                st.error("Please provide Fal.ai API Key.")
                st.stop()

            # 1. VISION IDENTIFICATION
            with st.spinner("🔍 Identifying jewelry features (Metal, Motif, Gems)..."):
                try:
                    id_handler = fal_client.submit(
                        VISION_ENDPOINT,
                        arguments={
                            "image_url": img_b64,
                            "prompt": "Describe this jewelry piece in detail. Identify the metal color, the main gemstone shape, the setting style, and any specific motifs (like floral, geometric, signet, etc)."
                        }
                    )
                    jewelry_info = id_handler.get()
                    description = jewelry_info['output']
                    st.success(f"**Identified:** {description}")
                    st.session_state['description'] = description
                except Exception as e:
                    st.error(f"Vision Analysis Failed: {e}")
                    st.stop()

            # 2. PREPARE PROMPTS BASED ON STYLES
            prompts_to_run = []
            
            if style_art_deco:
                base = f"Modify @image1: Transform this {description} into a vintage Art Deco style. REPLACE the setting with geometric patterns. Change metal to #FFD700 (Yellow Gold). Keep the center stone shape but add baguette side stones. High jewelry photography, 8k."
                prompts_to_run.extend([(base, "Art Deco") for _ in range(batch_size)])
            
            if style_minimal:
                base = f"Modify @image1: Transform this {description} into a modern Minimalist style. REMOVE intricate details. Change metal to #E6C2B4 (Rose Gold). Make the band smooth and thin. Focus on the center stone. Studio lighting."
                prompts_to_run.extend([(base, "Minimalist") for _ in range(batch_size)])
                
            if style_pavé:
                base = f"Modify @image1: Transform this {description} into a luxury Pavé style. REPLACE the band surface with micro-pavé diamonds. Change metal to #E5E4E2 (Platinum). Highly reflective, sparkle, cinematic lighting."
                prompts_to_run.extend([(base, "Pavé") for _ in range(batch_size)])

            # 3. GENERATE VARIATIONS
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            async def run_generations():
                tasks = []
                for p_text, style_name in prompts_to_run:
                    # Async submission to Flux Pro Edit
                    # Note: Flux Pro Edit takes "image_url" or "image_urls" depending on strict version.
                    # We use the standard pattern for Edit models.
                    tasks.append(fal_client.submit_async(
                        FLUX_EDIT_ENDPOINT,
                        arguments={
                            "prompt": p_text,
                            "image_url": img_b64, 
                            "strength": 0.75, # Strong edit for style transfer
                            "guidance_scale": 7.5
                        }
                    ))
                
                # Run in chunks to be safe
                chunk_size = 4
                completed = 0
                
                for i in range(0, len(tasks), chunk_size):
                    chunk = tasks[i:i+chunk_size]
                    batch_responses = await asyncio.gather(*chunk)
                    
                    for resp_handler in batch_responses:
                        try:
                            res = await resp_handler.get()
                            img_url = res['images'][0]['url']
                            # Download content
                            img_data = requests.get(img_url).content
                            # Find which prompt this belonged to (approximate mapping or simple index)
                            # For simplicity in this loop, we just append. 
                            # (Real prod code would map task-to-prompt more strictly)
                            results.append(img_data)
                        except Exception as e:
                            print(f"Gen Error: {e}")
                    
                    completed += len(chunk)
                    progress_bar.progress(min(completed / len(prompts_to_run), 1.0))
                    status_text.text(f"Generated {len(results)} / {len(prompts_to_run)} designs...")

            asyncio.run(run_generations())

            # 4. DISPLAY & DOWNLOAD
            if results:
                st.success("Redesign Complete!")
                
                # Gallery
                cols = st.columns(4)
                for idx, img_bytes in enumerate(results):
                    with cols[idx % 4]:
                        st.image(img_bytes, caption=f"Var #{idx+1}", use_container_width=True)
                
                # Zip
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    for idx, img_bytes in enumerate(results):
                        zf.writestr(f"redesign_{idx+1}.jpg", img_bytes)
                
                st.download_button(
                    "Download All Variations",
                    data=zip_buffer.getvalue(),
                    file_name="jewelbench_redesigns.zip",
                    mime="application/zip"
                )
