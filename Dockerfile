# 1. Base Image
FROM python:3.10-slim

# 2. System Dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ffmpeg \
      build-essential \
      curl \
    && rm -rf /var/lib/apt/lists/*

# 3. Set Working Directory
WORKDIR /app

# 4. Copy and Install Python Dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy Application Code
COPY . .

# 6. Environment Variables (override at runtime)
ENV FLASK_ENV=production \
    PORT=8080 \
    STORAGE_BUCKET=staff-471204.appspot.com

# 7. Expose Port
EXPOSE 8080

# 8. Entrypoint
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "4", "app:app"]
