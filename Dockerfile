FROM --platform=linux/amd64 python:3.10-slim-bullseye
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY process_pdfs.py .
COPY models/ ./models/
CMD ["python", "process_pdfs.py"]