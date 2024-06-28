# 使用官方的 Python 基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 复制依赖项定义文件到工作目录
COPY requirements.txt /app/

# 安装依赖项
RUN pip install --no-cache-dir -r requirements.txt

# 复制当前目录内容到工作目录
COPY . /app

# 暴露端口 8000
EXPOSE 8000

# 设置启动命令
CMD ["uvicorn", "fastApi:app", "--host", "0.0.0.0", "--port", "8000"]
