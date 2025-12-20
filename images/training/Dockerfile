FROM python:3.12.10-slim

RUN apt-get update && apt-get install -y build-essential

RUN pip install --upgrade pip
# Cache dependencies to speed up builds
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

WORKDIR /app

CMD ["python3"]