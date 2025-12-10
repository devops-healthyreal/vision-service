FROM python:3.10-slim

WORKDIR /app

# 환경 설정
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir --only-binary=:all: -r requirements.txt

COPY . .

EXPOSE 7200

CMD ["gunicorn", "--bind", "0.0.0.0:7005", "app:app"]