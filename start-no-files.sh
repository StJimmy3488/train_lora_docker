#!/bin/bash

# Base URL for remote files (Update this with the correct URL)

echo "Download complete!"

echo "starting nginx"

service nginx start 
# Start Flux Train UI
echo "Starting Flux Train UI..."
python /app/ai-toolkit/flux_train_ui.py &


# Keep the container running
wait
