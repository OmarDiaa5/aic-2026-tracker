FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download model checkpoint from public link
ARG MODEL_URL
RUN mkdir -p weights && wget -O weights/best_model.pt "${MODEL_URL}"

COPY . /app

CMD ["python", "inference.py", \
     "--data_dir", "/app/data", \
     "--checkpoint", "/app/weights/best_model.pt", \
     "--output", "/app/output/submission.csv"]
