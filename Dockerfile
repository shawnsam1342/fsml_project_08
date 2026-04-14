FROM python:3.10-slim

WORKDIR /app

# 🔹 Copy only requirements first (for caching)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# 🔹 Copy only necessary folders
COPY app/ app/
COPY src/ src/
COPY pipeline/ pipeline/
COPY models/ models/

EXPOSE 8000

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
