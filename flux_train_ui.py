import os
import sys
import uuid
import shutil
import json
import yaml
import logging
import time  # Import the time module

from PIL import Image
from dotenv import load_dotenv
import requests  # <-- NEW import for downloading URLs
import torch
import boto3
from botocore.exceptions import NoCredentialsError
from slugify import slugify
from transformers import AutoProcessor, AutoModelForCausalLM
import gradio as gr
from botocore.config import Config

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "ai-toolkit")
from toolkit.job import get_job

load_dotenv()
# Debug: Print loaded environment variables


# ----------------------------------------
# Setup Python's Built-in Logging
# ----------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,  # Changed to DEBUG logging level as suggested
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)


import boto3
from botocore.config import Config
import os

def get_s3_client():
    """Create S3-compatible client for Cloudflare R2"""
    s3_endpoint = os.getenv("S3_ENDPOINT")
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    region_name = os.getenv("AWS_REGION")

    if not s3_endpoint or not aws_access_key_id or not aws_secret_access_key:
        raise RuntimeError("Missing S3 credentials. Check .env file.")

    config = Config(
        signature_version="s3v4",  # ✅ Explicitly enforce Signature v4
        retries={"max_attempts": 3, "mode": "standard"}
    )

    return boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name,
        config=config
    )


def upload_directory_to_s3(local_dir, bucket_name, s3_prefix):
    """Uploads a directory to S3 while maintaining folder structure"""
    logger.info("Uploading directory '%s' to S3 bucket '%s' with prefix '%s'.",
                local_dir, bucket_name, s3_prefix)
    try:
        s3 = get_s3_client()
        for root, _, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, local_dir)
                s3_key = f"{s3_prefix}/{relative_path}"
                logger.debug("Uploading file '%s' to '%s'.", local_path, s3_key)
                s3.upload_file(local_path, bucket_name, s3_key)
        return True
    except NoCredentialsError:
        logger.error("AWS credentials not available.")
        return False
    except Exception as e:
        logger.exception("Error uploading to S3: %s", e)
        return False
    finally:
        # Clean up local directory regardless of upload success
        logger.debug("Removing local directory '%s'.", local_dir)
        shutil.rmtree(local_dir, ignore_errors=True)

# --------------------------------------------------------------------
# Helper: Resolve a single image input (dict/string/URL) to local path
# --------------------------------------------------------------------
def resolve_image_path(image_item):
    """
    Handle multiple possible 'image' input formats:
      - A Gradio file dict with 'name' key
      - A local path string
      - A URL (starting with http/https)
    Returns a local file path to be used by Pillow or shutil.copy
    """
    # If it's a Gradio-style dictionary
    if isinstance(image_item, dict) and "name" in image_item:
        return image_item["name"]

    # If it's a string, it might be a local path or a URL
    if isinstance(image_item, str):
        # Check if it looks like an http(s) URL
        if image_item.lower().startswith("http://") or image_item.lower().startswith("https://"):
            # Download to tmp_downloads
            os.makedirs("tmp_downloads", exist_ok=True)
            local_name = os.path.join("tmp_downloads", f"{uuid.uuid4()}.png")
            logger.debug("Downloading image from URL: %s --> %s", image_item, local_name)
            try:
                r = requests.get(image_item, stream=True)
                r.raise_for_status()
                with open(local_name, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                return local_name
            except Exception as ex:
                raise RuntimeError(f"Failed to download image from URL {image_item}: {ex}")

        # Otherwise assume it's a local path
        if not os.path.exists(image_item):
            raise ValueError(f"Image path does not exist: {image_item}")
        return image_item

    # If none of the above, it's invalid
    raise ValueError(f"Unsupported image type or format: {image_item}")

def process_images_and_captions(images, concept_sentence=None):
    """Process uploaded images and generate captions if needed"""
    logger.debug("Processing images for captioning. Number of images: %d", len(images))

    if len(images) < 2:
        raise gr.Error("Please upload at least 2 images")
    elif len(images) > 150:
        raise gr.Error("Maximum 150 images allowed")


    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16

    logger.info("Loading AutoModelForCausalLM from 'multimodalart/Florence-2-large-no-flash-attn'.")

    model = AutoModelForCausalLM.from_pretrained(
        "multimodalart/Florence-2-large-no-flash-attn",
        torch_dtype=torch_dtype,
        trust_remote_code=True
    ).to(device)

    processor = AutoProcessor.from_pretrained(
        "multimodalart/Florence-2-large-no-flash-attn",
        trust_remote_code=True
    )

    logger.debug("processor.post_process_generation => %s", processor.post_process_generation)
    captions = []
    try:
        for image_path in images:
            logger.debug("Processing image: %s", image_path)
            if isinstance(image_path, str):
                image = Image.open(image_path).convert("RGB")
                image.load()

            prompt = "<DETAILED_CAPTION>"
            inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(device, torch_dtype)

            logger.debug("Generating tokens with model.generate(...)")
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3
            )

            generated_text = processor.batch_decode(
                generated_ids,
                skip_special_tokens=False
            )[0]
            logger.debug("Generated text: %s", generated_text)

            try:
                logger.debug("Calling processor.post_process_generation(...)")
                parsed_answer = processor.post_process_generation(
                    generated_text,
                    task=prompt,
                    image_size=(image.width, image.height)
                )
            except Exception as ex:
                logger.error("Error calling post_process_generation: %s", ex, exc_info=True)
                raise

            logger.debug("Parsed answer: %s", parsed_answer)

            caption = parsed_answer["<DETAILED_CAPTION>"].replace(
                "The image shows ", ""
            )
            if concept_sentence:
                caption = f"{caption} [trigger]"
            captions.append(caption)
            logger.debug("Final caption for image: %s", caption)
    finally:
        logger.debug("Cleaning up model and processor from GPU/Memory.")
        model.to("cpu")
        del model
        del processor

    logger.info("Generated %d captions.", len(captions))
    return captions

def create_dataset(images, captions):
    """Create temporary dataset from images and captions"""
    destination_folder = f"tmp_datasets/{uuid.uuid4()}"
    logger.info("Creating a dataset in folder: %s", destination_folder)
    os.makedirs(destination_folder, exist_ok=True)

    jsonl_file_path = os.path.join(destination_folder, "metadata.jsonl")
    with open(jsonl_file_path, "a") as jsonl_file:
        for image_item, caption in zip(images, captions):
            # Convert 'image_item' to local path (could be URL or dict)
            local_path = resolve_image_path(image_item)
            logger.debug("Copying %s to dataset folder %s", local_path, destination_folder)

            new_image_path = shutil.copy(local_path, destination_folder)
            file_name = os.path.basename(new_image_path)
            data = {"file_name": file_name, "prompt": caption}
            jsonl_file.write(json.dumps(data) + "\n")
            logger.debug("Wrote to metadata.jsonl: %s", data)

    return destination_folder

def train_model(
    dataset_folder,
    lora_name,
    concept_sentence=None,
    steps=1000,
    lr=4e-4,
    rank=16,
    model_type="dev",
    low_vram=True,
    sample_prompts=None,
    advanced_options=None
):
    """Train the model and store exclusively in S3, returning a folder URL."""
    slugged_lora_name = slugify(lora_name)
    logger.info("Training LoRA model. Name: %s, Slug: %s", lora_name, slugged_lora_name)

    # Load default config
    logger.debug("Loading default config file: config/examples/train_lora_flux_24gb.yaml")
    with open("config/examples/train_lora_flux_24gb.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Update configuration
    config["config"]["name"] = slugged_lora_name
    process_block = config["config"]["process"][0]
    process_block.update({
        "model": {
            "low_vram": low_vram,
            "name_or_path": "black-forest-labs/FLUX.1-schnell" if model_type == "schnell" else "black-forest-labs/FLUX.1",
            "assistant_lora_path": "ostris/FLUX.1-schnell-training-adapter" if model_type == "schnell" else None
        },
        "train": {
            "skip_first_sample": True,
            "steps": int(steps),
            "lr": float(lr)
        },
        "network": {
            "linear": int(rank),
            "linear_alpha": int(rank)
        },
        "datasets": [{"folder_path": dataset_folder}],
        "save": {
            "output_dir": f"tmp_models/{slugged_lora_name}",
            "push_to_hub": False  # Disable Hugging Face push
        }
    })

    if concept_sentence:
        logger.debug("Setting concept_sentence (trigger_word) to '%s'.", concept_sentence)
        process_block["trigger_word"] = concept_sentence

    if sample_prompts:
        logger.debug("Sample prompts provided. Will enable sampling.")
        process_block["train"]["disable_sampling"] = False
        process_block["sample"].update({
            "sample_every": steps,
            "sample_steps": 28 if model_type == "dev" else 4,
            "prompts": sample_prompts
        })
    else:
        logger.debug("No sample prompts provided. Disabling sampling.")
        process_block["train"]["disable_sampling"] = True

    if advanced_options:
        logger.debug("Merging advanced_options YAML into config.")
        config["config"]["process"][0] = recursive_update(
            config["config"]["process"][0],
            yaml.safe_load(advanced_options)
        )

    # Save config
    config_path = f"tmp_configs/{uuid.uuid4()}-{slugged_lora_name}.yaml"
    logger.debug("Saving updated config to: %s", config_path)
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Run training
    logger.info("Retrieving job with config path: %s", config_path)
    job = get_job(config_path)
    logger.debug("job object => %s", job)
    if job is None:
        raise RuntimeError(f"get_job() returned None for config path: {config_path}. Please check your job definition.")

    s3_folder_url = None # Initialize s3_folder_url outside try block

    try: # Added try-except block around job.run() and job.cleanup()
        job_start_time = time.time() # Timing start for job.run()
        logger.info("Running job...")
        job.run()
        job_runtime = time.time() - job_start_time # Calculate job runtime
        logger.info(f"Job run completed in {job_runtime:.2f} seconds.") # Log job runtime

        cleanup_start_time = time.time() # Timing start for job.cleanup()
        logger.info("Cleaning up job...")
        job.cleanup()
        cleanup_runtime = time.time() - cleanup_start_time # Calculate cleanup runtime
        logger.info(f"Job cleanup completed in {cleanup_runtime:.2f} seconds.") # Log cleanup runtime


        # Upload to S3
        bucket_name = os.environ.get("S3_BUCKET")
        s3_domain = os.getenv("S3_DOMAIN", "https://r2.syntx.ai")
        local_model_dir = f"output/{slugged_lora_name}"


        if bucket_name and os.path.exists(local_model_dir):
            s3_prefix = f"loras/flux/{slugged_lora_name}"
            logger.info("Uploading trained model to S3: bucket=%s, prefix=%s", bucket_name, s3_prefix)
            upload_start_time = time.time() # Timing start for S3 upload
            logger.info("Starting S3 upload...")
            if upload_directory_to_s3(local_model_dir, bucket_name, s3_prefix):
                upload_runtime = time.time() - upload_start_time # Calculate upload runtime
                logger.info(f"S3 upload completed in {upload_runtime:.2f} seconds.") # Log upload runtime

                # Construct an HTTP-based “folder” URL on Timeweb S3
                s3_endpoint = os.environ.get("S3_ENDPOINT", "https://s3.timeweb.cloud").rstrip("/")
                s3_folder_url = f"{s3_domain}/{s3_prefix}/"
                logger.info("Model folder successfully uploaded to: %s", s3_folder_url)
            else:
                raise RuntimeError("upload_directory_to_s3 returned False indicating failure.") # Explicitly raise error if upload fails
        else:
            logger.warning("No S3_BUCKET set or local_model_dir does not exist. Skipping upload.")
            raise RuntimeError("S3 bucket not configured or model directory missing, cannot complete training.") # Raise error as S3 URL is essential

    except Exception as e_train_job: # Catch exceptions from job.run(), job.cleanup(), or S3 upload
        logger.error(f"Error during training job execution in train_model: {e_train_job}", exc_info=True)
        raise RuntimeError(f"Training job failed: {e_train_job}") from e_train_job # Re-raise to be caught by train_lora's except

    finally: # Cleanup always, regardless of training success or failure, and *before* returning
        # Cleanup
        logger.debug("Removing dataset folder: %s", dataset_folder)
        shutil.rmtree(dataset_folder, ignore_errors=True)
        logger.debug("Removing config file: %s", config_path)
        os.remove(config_path)


    if not s3_folder_url: # Check again after the try-except-finally block, in case S3 upload failed inside try
        msg = "Failed to obtain S3 folder URL after training, possibly due to upload failure or S3 configuration issues."
        logger.error(msg)
        raise RuntimeError(msg)

    return s3_folder_url


def recursive_update(d, u):
    """Recursively update nested dictionaries"""
    for k, v in u.items():
        if isinstance(v, dict) and v:
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def train_lora(
    images,
    lora_name,
    concept_sentence=None,
    steps=1000,
    lr=4e-4,
    rank=16,
    model_type="dev",
    low_vram=True,
    sample_prompts=None,
    advanced_options=None
):
    """Main training workflow with improved logging and error handling."""
    logger.info("Initiating train_lora with the following parameters: "
                "lora_name=%s, concept_sentence=%s, steps=%s, lr=%s, rank=%s, model_type=%s, low_vram=%s, sample_prompts=%s",
                lora_name, concept_sentence, steps, lr, rank, model_type, low_vram, sample_prompts)

    try:
        logger.info("1. Process images and captions.")
        captions = process_images_and_captions(images, concept_sentence)

        logger.info("2. Create dataset from images and captions.")
        dataset_folder = create_dataset(images, captions)

        logger.info("3. Train the model using train_model function.")
        folder_url = train_model(  # Call the modified train_model function
            dataset_folder,
            lora_name,
            concept_sentence,
            steps,
            lr,
            rank,
            model_type,
            low_vram,
            sample_prompts,
            advanced_options
        )
        logger.info("Training complete. Folder URL: %s", folder_url)
        return {"status": "success", "folder_url": folder_url}
    except RuntimeError as e_rt: # Catch RuntimeErrors specifically from train_model or other critical functions
        logger.error(f"RuntimeError in train_lora (likely training or S3 upload failure): {e_rt}")
        return {"status": "error", "message": str(e_rt)} # Return specific RuntimeError message
    except Exception as e: # Catch any other unexpected exceptions in train_lora itself
        logger.exception("Unexpected error in train_lora: %s", e) # Log full exception with traceback
        return {"status": "error", "message": "An unexpected error occurred during training. Check server logs for details."} # More generic message for client
    finally:
        # Cleanup temporary directories - moved to train_model's finally block to ensure cleanup even if train_model fails.
        logger.debug("Final cleanup (check if tmp dirs are removed by train_model's finally).")


# Gradio interface remains the same
demo = gr.Interface(
    fn=train_lora,
    inputs=[
        gr.File(file_count="multiple", label="Upload Images"),
        gr.Textbox(label="LoRA Name"),
        gr.Textbox(label="Trigger Word/Sentence (Optional)"),
        gr.Number(label="Steps", value=1000),
        gr.Number(label="Learning Rate", value=4e-4),
        gr.Number(label="LoRA Rank", value=16),
        gr.Radio(choices=["dev", "schnell"], value="dev", label="Model Type"),
        gr.Checkbox(label="Low VRAM Mode", value=True),
        gr.Textbox(label="Sample Prompts (Optional, comma-separated)"),
        gr.Code(label="Advanced Options (YAML)", language="yaml")
    ],
    outputs=gr.JSON(),
    title="FLUX LoRA Trainer API",
    description="Train a FLUX LoRA model with your images"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, max_threads=300, share=True, show_error=True)