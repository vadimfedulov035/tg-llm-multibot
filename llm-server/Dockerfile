FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

WORKDIR /app

RUN apt update && \
    apt install --no-install-recommends -y build-essential ca-certificates software-properties-common && \
    apt install --no-install-recommends -y python3.12 python3.12-dev python3.12-venv python3-pip python3-setuptools

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY ./llm-server/requirements.txt .
RUN python3.12 -m pip install --upgrade pip && \
    python3.12 -m pip install --no-cache-dir -r requirements.txt

COPY ./llm-server .

EXPOSE 8000

CMD ["uvicorn", "app:app" , "--host", "0.0.0.0", "--port", "8000", "--reload"]
