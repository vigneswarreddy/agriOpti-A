# Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
FROM python:3.10-slim

# Install system dependencies needed for some ML libraries (like OpenCV if used, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face requires running as a non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy the requirements file and install dependencies
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY --chown=user . /app

# Hugging Face Spaces expose port 7860
ENV PORT=7860

# Run the flask application using gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--timeout", "120", "app:app"]
