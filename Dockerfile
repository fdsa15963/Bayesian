# 使用官方 Python 輕量版作為基礎影像
FROM python:3.10-slim

# 設定工作目錄
WORKDIR /app

# 安裝系統相依套件 (若有需要編譯 numpy 等)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 複製相依套件清單並安裝
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用程式原始碼
COPY . .

# 暴露 FastAPI 預設埠口
EXPOSE 8000

# 啟動伺率器
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
