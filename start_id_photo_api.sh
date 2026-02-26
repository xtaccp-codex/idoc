#!/usr/bin/env bash

# 高级证件照 Web API 安装与启动脚本
# ==========================================
# 自动创建虚拟环境、安装依赖并启动 FastAPI 服务

set -e

echo "=========================================="
echo "  🚀 欢迎使用高级证件照生成器 Web API"
echo "=========================================="
echo ""

# 1. 检查 Python 环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未检测到 python3。请先安装 Python 3.8+。"
    exit 1
fi

echo "✅ 检测到 Python 引用: $(which python3)"
python3 --version

# 2. 检查或创建虚拟环境 (推荐做法，避免污染全局环境)
VENV_DIR="venv_id_photo"
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 正在创建虚拟环境: $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
else
    echo "✅ 虚拟环境已存在: $VENV_DIR"
fi

# 3. 激活虚拟环境
echo "🔄 激活虚拟环境..."
source "$VENV_DIR/bin/activate"

# 4. 安装/更新依赖包
echo "⏳ 正在安装必备依赖 (使用清华镜像源提速)..."
pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    fastapi \
    uvicorn \
    python-multipart \
    opencv-python \
    "numpy<2.0.0" \
    "pillow<11.0.0" \
    'rembg[cpu]'

echo "✅ 依赖安装完成！"
echo ""

# 5. 启动服务
echo "=========================================="
echo "  🌐 准备启动 API 服务"
echo "  👉 接口文档: http://127.0.0.1:8000/docs"
echo "  ⏹  按 Ctrl+C 停止服务"
echo "=========================================="
echo ""
echo "⏳ 首次启动时，AI 模型需要预热与底层编译(Numba JIT)，大约需要 15~30 秒，请耐心等待..."

# 运行 FastAPI
python api_server.py
