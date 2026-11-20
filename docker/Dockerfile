
# 1. Start with a lightweight Python image
FROM python:3.11-slim

# 2. Set the working directory
WORKDIR /app

# 3. Prevent Python from writing bytecode
ENV PYTHONDONTWRITEBYTECODE=1

# 4. Display Python logs immediately
ENV PYTHONUNBUFFERED=1

# 5. Install CPU-only PyTorch
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision

# 6. Copy dependency file
COPY requirements.txt .

# 7. Install application dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 8. Copy application code
COPY src/satellite_classifier/ ./satellite_classifier/

# 9. Copy trained model checkpoint
COPY models/best_model.pth ./models/best_model.pth

# 10. Expose API port
EXPOSE 8000

# 11. Start FastAPI
CMD ["uvicorn", "satellite_classifier.api:app", "--host", "0.0.0.0", "--port", "8000"]