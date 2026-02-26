import os
import secrets
import shutil
import tempfile
import asyncio
import ctypes
import gc
import logging
from pathlib import Path
from functools import partial

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import Response
import uvicorn

# 导入生成证件照的核心函数
from generate_id_photo_advanced import (
    generate_id_photo,
    parse_bg_color,
    parse_size,
    validate_image_file,
    clear_rembg_session
)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Advanced ID Photo Generator API",
    description="商业级证件照生成器 HTTP 服务，支持 AI 精确抠图与边缘羽化",
    version="2.0",
    openapi_url="/openapi.json",
    docs_url="/docs",
    servers=[
        {"url": "/idoc-api", "description": "Production environment via Nginx"}
    ]
)

# 确保有个临时存储上传文件的工作区
TEMP_DIR = Path(tempfile.gettempdir()) / "id_photo_api"

def init_temp_dir():
    """初始化临时目录，并清理上次残留的文件"""
    if TEMP_DIR.exists():
        # 清理该目录下所有的 input_ 和 output_ 文件
        for item in TEMP_DIR.glob("*"):
            if item.is_file():
                try:
                    item.unlink()
                except:
                    pass
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"已初始化并清理临时目录: {TEMP_DIR}")

init_temp_dir()

# 推理信号量：同一时刻最多只允许 1 个 AI 推理任务运行
# 在 2G 内存服务器上，并发推理会直接 OOM 炸服
_inference_semaphore = asyncio.Semaphore(1)

# Linux 下加载 glibc，用于强制释放内存回给操作系统
try:
    _libc = ctypes.CDLL("libc.so.6")
    def _force_release_memory():
        """Python 的 gc.collect() 只回收对象，但不会把内存还给 OS。
        调用 glibc 的 malloc_trim(0) 可以强制把空闲的内存页归还给操作系统。"""
        gc.collect()
        _libc.malloc_trim(0)
        logger.info("已强制释放内存回给操作系统")
except OSError:
    # macOS / Windows 上没有 libc.so.6，回退为纯 gc
    def _force_release_memory():
        gc.collect()
        logger.info("已执行 gc.collect()（非 Linux 环境，跳过 malloc_trim）")

def cleanup_files(*file_paths: Path):
    """清理临时文件"""
    for file_path in file_paths:
        try:
            if file_path and file_path.exists():
                file_path.unlink()
        except Exception as e:
            logger.warning(f"清理文件 {file_path} 失败: {e}")


def _blocking_generate(input_path, output_path, bg_color, output_size, skip_quality_check):
    """
    封装同步阻塞的核心生成逻辑，供线程池调用。
    这样就不会堵塞 FastAPI 的主事件循环，让服务器在处理一张图片时仍然能响应其他请求。
    """
    return generate_id_photo(
        input_path=input_path,
        output_path=output_path,
        bg_color=bg_color,
        output_size=output_size,
        skip_quality_check=skip_quality_check,
        verbose=False
    )


@app.post("/generate", summary="生成证件照")
async def generate_photo_endpoint(
    file: UploadFile = File(..., description="上传的原始人物照片 (JPG/PNG)"),
    bg_color: str = Form("red", description="背景颜色: red/blue/white/gray 或 #RRGGBB"),
    size: str = Form("2inch", description="输出尺寸: 1inch/2inch/passport 等，或 WxH"),
    skip_quality_check: bool = Form(False, description="是否跳过画质太差的阻塞拦截"),
):
    """
    上传照片并生成标准证件照，返回生成的图片文件流。
    """
    
    # === 参数格式和合法性前置拦截 ===
    try:
        parsed_bg_color = parse_bg_color(bg_color)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    try:
        parsed_size = parse_size(size)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    # === 读入与保存临时文件 ===
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ['.jpg', '.jpeg', '.png', '.webp']:
        raise HTTPException(status_code=400, detail="不支持的图片格式，仅支持 JPG/PNG/WEBP")

    # 生成安全的随机文件名防覆盖
    unique_id = secrets.token_hex(8)
    input_tmp_path = TEMP_DIR / f"input_{unique_id}{suffix}"
    output_tmp_path = TEMP_DIR / f"output_{unique_id}.jpg"

    try:
        with open(input_tmp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # === 【并发保护】同一时间只允许一个推理任务，防止 2G 服务器 OOM ===
        async with _inference_semaphore:
            loop = asyncio.get_event_loop()
            result_bgr = await loop.run_in_executor(
                None,
                partial(
                    _blocking_generate,
                    input_path=str(input_tmp_path),
                    output_path=str(output_tmp_path),
                    bg_color=parsed_bg_color,
                    output_size=parsed_size,
                    skip_quality_check=skip_quality_check,
                )
            )
            
            if result_bgr is None:
                raise HTTPException(
                    status_code=422, 
                    detail="证件照生成失败，可能是质量检测未通过，或 AI 背景去除、人脸裁切异常。请检查服务器日志或尝试开启 skip_quality_check=True"
                )

            # 先读取生成的图片到内存
            with open(output_tmp_path, "rb") as f:
                image_bytes = f.read()

            # 在信号量保护内完成清理
            del result_bgr
            # clear_rembg_session()  # 已有 8G Swap，不再释放模型，以换取极速响应速度
            _force_release_memory()  # 依然执行 gc + malloc_trim 归还非模型内存给系统

        return Response(
            content=image_bytes,
            media_type="image/jpeg",
            headers={
                "Content-Disposition": f'attachment; filename="id_photo_{bg_color}_{size}.jpg"'
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器处理未知异常: {str(e)}")
    finally:
        # 【终极修复】无论成功与否，必定清理临时文件
        cleanup_files(input_tmp_path, output_tmp_path)


if __name__ == "__main__":
    print(f"🚀 启动证件照 Web API 服务...")
    print(f"📚 访问 Swagger 文档调试: http://127.0.0.1:8000/docs")
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000)
