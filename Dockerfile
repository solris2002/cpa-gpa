# Sử dụng image Python cơ bản
FROM python:3.10-slim

# Cài các package hệ thống nếu cần (ví dụ: git, ffmpeg, build-essential,...)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Tạo thư mục app
WORKDIR /app

# Copy toàn bộ source code vào container
COPY . /app

# Cài dependencies Python
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# Cổng để Streamlit chạy (default 7860)
EXPOSE 7860

# Chạy ứng dụng (Streamlit app)
CMD ["streamlit", "run", "main.py", "--server.port=7860", "--server.address=0.0.0.0"]
