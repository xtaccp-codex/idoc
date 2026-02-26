import os
import secrets
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, root_validator
import uvicorn

# 导入生成证件照的核心函数
from generate_id_photo_advanced import (
    generate_id_photo,
    parse_bg_color,
    parse_size,
    validate_image_file
)

app = FastAPI(
    title="Advanced ID Photo Generator API",
    description="商业级证件照生成器 HTTP 服务，支持 AI 精确抠图与边缘羽化",
    version="2.0",
    root_path="/idoc-api",  # 用于配合 Nginx 的 /idoc-api/ 转发，让 Swagger 接口测试页面上的路径不变错
)

# 确保有个临时存储上传文件的工作区
TEMP_DIR = Path(tempfile.gettempdir()) / "id_photo_api"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

def cleanup_files(*file_paths: Path):
    """后台任务清理临时文件"""
    for file_path in file_paths:
        try:
            if file_path and file_path.exists():
                file_path.unlink()
        except Exception as e:
            print(f"清理文件 {file_path} 失败: {e}")


@app.post("/generate", summary="生成证件照")
async def generate_photo_endpoint(
    background_tasks: BackgroundTasks,
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
            
        # === 执行核心生成逻辑 ===
        # 调用已经封装好的商业级能力 (注意：verbose=False 关闭打印防刷屏)
        result_bgr = generate_id_photo(
            input_path=str(input_tmp_path),
            output_path=str(output_tmp_path),
            bg_color=parsed_bg_color,
            output_size=parsed_size,
            skip_quality_check=skip_quality_check,
            verbose=False
        )
        
        if result_bgr is None:
            # 失败可能由画质差被拦截、人脸未检出异常等原因引起
            raise HTTPException(
                status_code=422, 
                detail="证件照生成失败，可能是质量检测未通过，或 AI 背景去除、人脸裁切异常。请检查服务器日志或尝试开启 skip_quality_check=True"
            )

        # === 安排后台清理任务并在前台返回图片 ===
        # 在返回 HTTP 响应给客户端之后，让 FastAPI 异步删去临时存放的图
        background_tasks.add_task(cleanup_files, input_tmp_path, output_tmp_path)
        
        return FileResponse(
            path=str(output_tmp_path),
            media_type="image/jpeg",
            filename=f"id_photo_{bg_color}_{size}.jpg"
        )
        
    except HTTPException:
        # 直接透传 HTTP 错误，清理产生的仅有的原始输入
        cleanup_files(input_tmp_path)
        raise
    except Exception as e:
        cleanup_files(input_tmp_path, output_tmp_path)
        raise HTTPException(status_code=500, detail=f"服务器处理未知异常: {str(e)}")


if __name__ == "__main__":
    print(f"🚀 启动证件照 Web API 服务...")
    print(f"📚 访问 Swagger 文档调试: http://127.0.0.1:8000/docs")
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
