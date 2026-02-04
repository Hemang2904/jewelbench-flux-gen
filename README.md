# Flux 2.0 Jewelry Variation Generator

This Streamlit application allows for bulk generation of jewelry design variations using the Flux 2.0 API. It supports text, image, or combined inputs to create high-quality, 3/4 angle jewelry concepts suitable for 3D modeling references.

## Features

*   **Multi-modal Input:** Accept Text prompts, Image references, or both simultaneously.
*   **Bulk Generation:** Generate 25, 50, 75, or 100 images in a single batch.
*   **Consistency Control:** Maintains input image angle (optimized for 3/4 view) and stylistic theme.
*   **Deduplication:** Hashing mechanism to prevent saving exact duplicate images.
*   **Zip Download:** Easily download all generated variations in a structured zip file.

## Setup

1.  Clone this repository.
2.  Install requirements:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the app:
    ```bash
    streamlit run app.py
    ```

## Configuration

Enter your Flux 2.0 API Key in the sidebar to begin.
