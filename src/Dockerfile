FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/
COPY data/ /app/data/
COPY assets/ /app/assets/
COPY .streamlit/ /app/.streamlit/

RUN mkdir -p /app/models/runs

EXPOSE 8501

CMD ["streamlit", "run", "home.py", "--server.port=8501", "--server.address=0.0.0.0"]
