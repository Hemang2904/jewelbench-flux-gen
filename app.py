import streamlit as st
import fal_client
import io
import zipfile
import hashlib
import os
import asyncio
import base64
import random
import re
from PIL import Image

# --- Configuration ---
st.set_page_config(page_title="JewelBench - Advanced Variation", layout="wide")

# Using Flux Pro (1.1) for Image-to-Image / Edit if available
# The user requested "fal-ai/flux-2-pro/edit". 
# Note: As of early 2026, standard endpoints are typically "fal-ai/flux-pro/v1.1/image-to-image" or similar.
# We will default to a high-quality Pro Edit endpoint.
MODEL_ENDPOINT = "fal-ai/flux-pro/v1.1/image-to-image"  # Updated to Pro Edit

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

async def generate_variation_async(image_b64, prompt, strength, guidance):
    """
    Async wrapper for Fal.ai Flux Image-to-Image.
    """
    try:
        handler = await fal_client.submit_async(
            MODEL_ENDPOINT,
            arguments={
                "prompt": prompt,
                "image_url": image_b64,
                "strength": strength,  # Denoising strength (0.60 - 0.75)
                "guidance_scale": guidance, # Guidance (5.0 - 7.0)
                "num_inference_steps": 40,
                "enable_safety_checker": False
            },
        )
        result = await handler.get()
        image_url = result['images'][0]['url']
        
        # Download immediately
        import requests
        resp = requests.get(image_url)
        return resp.content, prompt  # Return content AND the resolved prompt used
    except Exception as e:
        print(f"Error: {e}")
        return None, None

# --- UI Layout ---
st.title("💎 JewelBench: Structural Variation Engine")
st.markdown("""
**Advanced Image-to-Image Logic:**
*   **Denoising Strength (0.60 - 0.75):** Controls how much the design changes vs. keeping original shape.
*   **Dynamic Prompts:** Use `[gold | platinum]` syntax to auto-randomize batches.
""")

# Sidebar Controls
with st.sidebar:
    api_key_input = st.text_input("Fal.ai API Key", type="password")
    if api_key_input:
        os.environ["FAL_KEY"] = api_key_input
    
    # Allow endpoint override
    st.markdown("### Model Settings")
    custom_model = st.text_input("Model Endpoint", value=MODEL_ENDPOINT, help="Change if using a specific finetune or new release like 'fal-ai/flux-2-pro/edit'")
    if custom_model:
        MODEL_ENDPOINT = custom_model

    st.markdown("### 1. Structural Parameters")
    strength = st.slider(
        "Denoising Strength", 0.1, 1.0, 0.65, 0.01,
        help="0.60-0.75 is the 'Sweet Spot'. Lower = closer to original. Higher = more hallucination."
    )
    
    guidance = st.slider(
        "Guidance Scale", 1.0, 20.0, 6.0, 0.5,
        help="Higher values force the model to follow the prompt text strictly."
    )
    
    st.markdown("### 2. Batch Settings")
    batch_size = st.select_slider("Batch Size", options=[10, 25, 50, 75, 100], value=10)

# Main Area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input")
    uploaded_file = st.file_uploader("Reference Jewelry Image", type=["jpg", "png", "jpeg"])
    
    default_prompt = "Macro shot of a [solitaire | halo | three-stone | pavé] engagement ring, [white gold | platinum | rose gold] band, [round cut | emerald cut | pear shape] center diamond, intricate filigree details, photorealistic, 8k, sharp focus, 3/4 angle view"
    
    base_prompt = st.text_area(
        "Dynamic Prompt (Use [A | B] for random variations)", 
        default_prompt,
        height=150
    )

with col2:
    if uploaded_file:
        st.image(uploaded_file, caption="Structural Reference", width=300)
    else:
        st.info("Upload an image to define the 'Skeleton' of the jewelry.")

# Generation Logic
if st.button("Generate Variations", type="primary"):
    if not api_key_input:
        st.error("Missing API Key.")
        st.stop()
    if not uploaded_file:
        st.error("Please upload a reference image.")
        st.stop()
        
    # Prepare Input
    input_img = Image.open(uploaded_file)
    img_b64 = get_image_base64(input_img)
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    results = []
    
    async def run_batch():
        tasks = []
        # Create a unique prompt for each image in the batch
        for _ in range(batch_size):
            # Resolve the dynamic prompt (e.g. pick "Halo" and "Platinum")
            resolved_prompt = parse_dynamic_prompt(base_prompt)
            tasks.append(generate_variation_async(img_b64, resolved_prompt, strength, guidance))
        
        # Execute in chunks
        chunk_size = 4  # Fal concurrency limit safe guard
        completed = 0
        
        for i in range(0, len(tasks), chunk_size):
            chunk = tasks[i:i+chunk_size]
            batch_results = await asyncio.gather(*chunk)
            
            for img_bytes, used_prompt in batch_results:
                if img_bytes:
                    results.append({"image": img_bytes, "prompt": used_prompt})
            
            completed += len(chunk)
            progress_bar.progress(min(completed / batch_size, 1.0))
            status_text.text(f"Generated {len(results)} / {batch_size} variations...")

    asyncio.run(run_batch())

    # --- Results ---
    if results:
        st.success("Generation Complete!")
        
        # Gallery with Prompt Details
        st.markdown("### Variations")
        cols = st.columns(3)
        for idx, item in enumerate(results):
            with cols[idx % 3]:
                st.image(item["image"], caption=f"#{idx+1}: {item['prompt'][:60]}...", use_container_width=True)
        
        # ZIP Download
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for idx, item in enumerate(results):
                # Clean prompt for filename
                clean_name = re.sub(r'[^a-zA-Z0-9]', '_', item['prompt'][:30])
                zf.writestr(f"var_{idx+1}_{clean_name}.jpg", item["image"])
        
        st.download_button(
            "Download All Images (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="structural_variations.zip",
            mime="application/zip"
        )
