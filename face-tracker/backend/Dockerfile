FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# System dependencies for OpenCV, YuNet, SFace
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download YuNet and SFace models
RUN mkdir -p /app/models && \
    wget -q -O /app/models/face_detection_yunet_2023mar.onnx \
      https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx && \
    wget -q -O /app/models/face_recognition_sface_2021dec.onnx \
      https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx

COPY . .

RUN mkdir -p /app/face_images /app/videos

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]