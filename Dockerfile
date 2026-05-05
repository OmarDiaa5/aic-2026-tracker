FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt kagglehub

# Download model checkpoint from Kaggle Models
RUN python -c "import kagglehub, shutil, glob, os; \
    path = kagglehub.model_download('omardiaa05/a-eye-model/pyTorch/default'); \
    pt_file = glob.glob(os.path.join(path, '*.pt'))[0]; \
    os.makedirs('weights', exist_ok=True); \
    shutil.copy(pt_file, 'weights/best_model.pt')"

COPY . /app

CMD ["python", "inference.py", \
     "--data_dir", "/app/data", \
     "--checkpoint", "/app/weights/best_model.pt", \
     "--output", "/app/output/submission.csv"]
