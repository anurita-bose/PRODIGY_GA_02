"""
PRODIGY_GA_02 — Image Generation with Pre-trained Models
Task: Utilize pre-trained generative models (Stable Diffusion) to create
images from text prompts.
"""

import os
import shutil
import torch
from diffusers import StableDiffusionPipeline


# --- Step 1: Confirm GPU ---
print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- Step 2: Load Stable Diffusion pipeline ---
MODEL_ID = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
)
pipe = pipe.to(DEVICE)
pipe.set_progress_bar_config(disable=True)


# --- Step 3: Prompts to generate ---
PROMPTS = [
    "a futuristic city skyline at sunset, digital art",
    "a photorealistic portrait of an old fisherman, dramatic lighting",
    "a cozy cabin in a snowy forest, oil painting style",
    "an astronaut riding a horse on mars, highly detailed",
]


def generate_core_images(output_dir="outputs"):
    """Generate one image per core prompt using default settings."""
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    for i, prompt in enumerate(PROMPTS):
        print(f"Generating image {i + 1}/{len(PROMPTS)}: {prompt}")
        image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
        filename = os.path.join(output_dir, f"image_{i + 1}.png")
        image.save(filename)
        results[prompt] = filename
    return results


def generate_guidance_scale_comparison(test_prompt, output_dir="outputs/guidance_comparison"):
    """
    guidance_scale controls how strictly the model follows the prompt.
    Low = more creative/random, High = more literal prompt-adherence.
    """
    os.makedirs(output_dir, exist_ok=True)
    guidance_scales = [3, 7.5, 15]
    for gs in guidance_scales:
        print(f"Generating with guidance_scale={gs}")
        image = pipe(test_prompt, num_inference_steps=50, guidance_scale=gs).images[0]
        image.save(os.path.join(output_dir, f"gs_{gs}.png"))


def generate_steps_comparison(test_prompt, output_dir="outputs/steps_comparison"):
    """
    num_inference_steps controls how many denoising iterations run.
    More steps generally = higher quality, at the cost of speed.
    """
    os.makedirs(output_dir, exist_ok=True)
    steps_list = [10, 25, 50]
    for steps in steps_list:
        print(f"Generating with num_inference_steps={steps}")
        image = pipe(test_prompt, num_inference_steps=steps, guidance_scale=7.5).images[0]
        image.save(os.path.join(output_dir, f"steps_{steps}.png"))


if __name__ == "__main__":
    core_results = generate_core_images()

    test_prompt = "a futuristic city skyline at sunset, digital art"
    generate_guidance_scale_comparison(test_prompt)
    generate_steps_comparison(test_prompt)

    # Zip everything for easy download (useful on Colab)
    shutil.make_archive("PRODIGY_GA_02_outputs", "zip", "outputs")
    print("\nAll images generated. Outputs saved to ./outputs and zipped as PRODIGY_GA_02_outputs.zip")
