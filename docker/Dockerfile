FROM python:3.11-slim

WORKDIR /app

# Install required libraries
RUN apt-get update && \
    # for opencv
    apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements/requirements-demo.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy config files
COPY params.yaml .
# Copy src files
COPY src src
# Copy example images
ENV IMG_DIR=data/datasets/serie02_plus_serie01_png
COPY ${IMG_DIR}/30057.png ${IMG_DIR}/30057.png
COPY ${IMG_DIR}/30014_2.png ${IMG_DIR}/30014_2.png
COPY ${IMG_DIR}/30143.png ${IMG_DIR}/30143.png
COPY ${IMG_DIR}/10004.png ${IMG_DIR}/10004.png
COPY ${IMG_DIR}/10012.png ${IMG_DIR}/10012.png
COPY ${IMG_DIR}/10035.png ${IMG_DIR}/10035.png
# Copy model
COPY out/export/model.onnx out/export/model.onnx

EXPOSE 7860

# Run the app
CMD ["python", "-m", "src.serve"]
