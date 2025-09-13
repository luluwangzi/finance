FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# System deps (curl for yfinance fallback, fonts for plots)
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    curl ca-certificates fonts-dejavu-core && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["bash", "-lc", "streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true"]

