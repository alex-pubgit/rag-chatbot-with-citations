FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY sample_docs ./sample_docs
RUN useradd --create-home app && chown -R app /app
USER app

# The image ships with the sample corpus already indexed, so it answers on first start.
RUN ragbot ingest sample_docs

EXPOSE 8000
CMD ["ragbot", "serve", "--host", "0.0.0.0", "--port", "8000"]
