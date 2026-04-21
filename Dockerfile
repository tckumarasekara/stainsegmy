# Use CUDA-enabled PyTorch image
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    click==8.3.1 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    rich==14.3.3 \
    tifffile==2025.5.10 \
    imagecodecs==2025.3.30 \
    captum==0.8.0 \
    pytorch_lightning==2.6.1 \
    torchvision==0.16.1

# Create directory for downloaded models
RUN mkdir -p /app/models

# Default command
ENTRYPOINT ["python", "cli_pred.py"]

#docker run --rm --gpus all -v $(pwd):/data -v $(pwd):/output stainsegmy -i /data/2751_CRC027-rack-01-well-A01-roi-001.tif -o /output --cuda --architecture U-NeXt