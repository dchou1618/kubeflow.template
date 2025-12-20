FROM python:3.12.10-slim

RUN pip install --upgrade pip

# Cache dependencies to speed up builds
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

WORKDIR /app

CMD ["python3"]