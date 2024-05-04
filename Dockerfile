FROM python:3.11-buster

ENV PROJECT_DIR llm
WORKDIR /${PROJECT_DIR}
COPY ./requirements.txt /${PROJECT_DIR}/

RUN pip install --upgrade pip && \
  pip install --no-cache-dir -r requirements.txt

COPY ./src/ /${PROJECT_DIR}/src
