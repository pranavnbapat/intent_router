# Dockerfile
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NLTK_DATA=/usr/local/nltk_data

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
      tini curl libgomp1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN useradd -m -u 10001 appuser

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Fetch NLTK stopwords at build time into a world-readable location
RUN mkdir -p "$NLTK_DATA" \
 && python -c "import nltk; nltk.download('stopwords', download_dir='$NLTK_DATA')"

# App code
COPY . .

ENV PYTHONPATH=/app

# Run as non-root
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000","--proxy-headers","--forwarded-allow-ips","*","--timeout-keep-alive","120"]
