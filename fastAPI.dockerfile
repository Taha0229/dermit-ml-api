FROM python:3.10-slim
ENV OPENAI_API_KEY="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"
ENV PINECONE_API_KEY="a5af7243-73e9-44f1-8869-4583c167a243"
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .


