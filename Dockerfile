# 1. Start from an official, lightweight Python image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy requirements first and install (Docker caches this layer so future builds are super fast)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy your project code, frontend, and saved model into the container
COPY . .

# 5. Expose port 8000 for FastAPI
EXPOSE 8000

# 6. Start the FastAPI server when the container launches
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]