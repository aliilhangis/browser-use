FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip setuptools && \
    /opt/venv/bin/pip install -r requirements.txt

COPY . .

CMD ["/opt/venv/bin/python", "main.py"]
