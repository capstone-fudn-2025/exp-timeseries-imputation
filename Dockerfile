FROM python:3.11.11-slim-bookworm

ENV PYTHONUNBUFFERED=1

# Install required packages
RUN pip install poetry && poetry config virtualenvs.create false
COPY pyproject.toml poetry.lock ./
RUN poetry install --sync --without dev