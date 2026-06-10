# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder
WORKDIR /app

COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-api.txt

# Pre-download NLTK corpora so the runtime image never needs internet access
RUN python -c "\
import nltk; \
nltk.download('stopwords', quiet=True); \
nltk.download('wordnet', quiet=True); \
nltk.download('omw-1.4', quiet=True)"

# ── Stage 2: api runtime ──────────────────────────────────────────────────────
FROM python:3.10-slim AS api
WORKDIR /app

COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /root/nltk_data /root/nltk_data

COPY src/ ./src/
COPY models/ ./models/
COPY settings/ ./settings/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ── Stage 3: frontend builder ──────────────────────────────────────────────────
FROM node:20-slim AS frontend-builder
WORKDIR /frontend

COPY frontend/package*.json ./
RUN npm ci --prefer-offline

COPY frontend/ ./
RUN npm run build

# ── Stage 4: frontend runtime ──────────────────────────────────────────────────
FROM nginx:alpine AS frontend

COPY --from=frontend-builder /frontend/dist/frontend/browser /usr/share/nginx/html
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
