#!/bin/bash

# Base URL for remote files
BASE_URL="https://syntxai.net/sd/FLUX.1-dev/"

# Destination directory
DEST_DIR="/app/ai-toolkit/FLUX.1-dev"

# Log file
LOG_FILE="/app/logs/download.log"
mkdir -p /app/logs
echo "Starting download process" > "$LOG_FILE"

# List of files to download
FILES=(
    "dev_grid.jpg"
    "ae.safetensors"
    "LICENSE.md"
    "model_index.json"
    "README.md"
    "scheduler/scheduler_config.json"
    "text_encoder/config.json"
    "text_encoder/model.safetensors"
    "text_encoder_2/config.json"
    "text_encoder_2/model-00001-of-00002.safetensors"
    "text_encoder_2/model-00002-of-00002.safetensors"
    "text_encoder_2/model.safetensors.index.json"
    "tokenizer/merges.txt"
    "tokenizer/special_tokens_map.json"
    "tokenizer/tokenizer_config.json"
    "tokenizer/vocab.json"
    "tokenizer_2/special_tokens_map.json"
    "tokenizer_2/spiece.model"
    "tokenizer_2/tokenizer.json"
    "tokenizer_2/tokenizer_config.json"
    "transformer/config.json"
    "transformer/diffusion_pytorch_model-00001-of-00003.safetensors"
    "transformer/diffusion_pytorch_model-00002-of-00003.safetensors"
    "transformer/diffusion_pytorch_model-00003-of-00003.safetensors"
    "transformer/diffusion_pytorch_model.safetensors.index.json"
    "vae/config.json"
    "vae/diffusion_pytorch_model.safetensors"
)

# Function to download files with verification
download_model() {
    local url="$1"
    local dest="$2"

    # Ensure directory exists
    mkdir -p "$(dirname "$dest")"

    # Check if file exists & delete empty or corrupted files
    if [ -f "$dest" ]; then
        echo "Checking integrity of existing file: $dest" | tee -a "$LOG_FILE"

        expected_size=$(curl -sI "$url" | grep -i Content-Length | awk '{print $2}' | tr -d '\r')
        actual_size=$(stat -c%s "$dest")

        if [[ -z "$expected_size" || "$actual_size" -lt "$expected_size" ]]; then
            echo "⚠️ Corrupted file detected, re-downloading: $dest" | tee -a "$LOG_FILE"
            rm -f "$dest"
        else
            echo "✅ File is valid: $dest ($actual_size bytes)" | tee -a "$LOG_FILE"
            return 0
        fi
    fi

    # Download file with retries
    echo "⬇️ Downloading: $dest" | tee -a "$LOG_FILE"
    wget --tries=3 --timeout=30 --continue --progress=bar:force:noscroll -O "$dest" "$url"

    # Verify download
    actual_size=$(stat -c%s "$dest")
    if [[ "$actual_size" -lt "$expected_size" ]]; then
        echo "❌ Failed to download properly: $dest" | tee -a "$LOG_FILE"
        rm -f "$dest"
        exit 1
    fi

    echo "✅ Successfully downloaded: $dest ($actual_size bytes)" | tee -a "$LOG_FILE"
}

# Create necessary directories
echo "📂 Creating directory structure..."
mkdir -p "$DEST_DIR"
for file in "${FILES[@]}"; do
    mkdir -p "$DEST_DIR/$(dirname "$file")"
done

# Download all required files
echo "🚀 Starting download process..."
for file in "${FILES[@]}"; do
    download_model "$BASE_URL/$file" "$DEST_DIR/$file"
done

echo "✅ All downloads completed successfully!"



# Download all required files

echo "starting nginx"

service nginx start 
# Start Flux Train UI
echo "Starting Flux Train UI..."
python /app/ai-toolkit/flux_train_ui.py &


# Keep the container running
wait
