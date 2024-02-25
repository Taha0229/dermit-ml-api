FROM python:3.10-slim

WORKDIR /app

RUN conda install onnxruntime -c conda-forge -y
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
