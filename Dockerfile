FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src /app/src
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
