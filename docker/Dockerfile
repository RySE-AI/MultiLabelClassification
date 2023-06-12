# app/Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY model /app/model

RUN apt-get update && apt-get upgrade -y && apt-get install --no-install-recommends -y \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/RySE-AI/MultiLabelClassification.git 

RUN python -m pip install --upgrade pip && \
    pip install \
    torch==2.0.1 \
    torchvision==0.15.2 \
    streamlit==1.23.1 \
    lightning==2.0.2 \
    opencv-python \
    seaborn


EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "MultiLabelClassification/scripts/mtg_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]