#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级证件照生成器 — 商业级版本
============================

功能特性:
  ✅ AI 精确抠图（rembg）+ 边缘羽化处理
  ✅ 智能人脸检测 + 胸部裁剪
  ✅ 自定义背景颜色（预设 red/blue/white/gray 或十六进制 #RRGGBB）
  ✅ 标准证件照尺寸输出（一寸、二寸、护照等）
  ✅ 图片质量检测（分辨率、清晰度、亮度）
  ✅ 批量处理支持（多线程并发）
  ✅ 完整异常处理和日志记录
  ✅ 商业级代码质量（类型提示、文档字符串）

依赖安装:
  pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python numpy pillow 'rembg[cpu]'

单张处理:
  python3 generate_id_photo_advanced.py -i photo.jpg
  python3 generate_id_photo_advanced.py -i photo.jpg -o output.jpg -bg blue
  python3 generate_id_photo_advanced.py -i photo.jpg -bg white -s 1inch
  python3 generate_id_photo_advanced.py -i photo.jpg -bg "#0066CC" -s 295x413

批量处理:
  python3 generate_id_photo_advanced.py -i photos/ --batch
  python3 generate_id_photo_advanced.py -i photos/ --batch -o output/ -bg blue --workers 8

作者: AI Assistant
版本: 2.0 (商业级)
更新: 2026-02-26
"""
import argparse
import sys
import logging
from pathlib import Path
import threading
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from PIL import Image

try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    print("⚠️  rembg 未安装，请执行: pip3 install 'rembg[cpu]'")

from PIL import ImageOps

# ── 配置日志 ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 全局缓存 rembg session
_REMBG_SESSION = None

def get_rembg_session():
    """获取缓存的 rembg session (单例模式)"""
    global _REMBG_SESSION
    if _REMBG_SESSION is None:
        try:
            _REMBG_SESSION = new_session("u2net")
        except Exception as e:
            logger.error(f"加载 rembg 模型失败: {e}")
    return _REMBG_SESSION

# ── 预设配置 ──────────────────────────────────────────────


PRESET_COLORS = {
    "red":   (238,  28,  37),   # 中国标准证件照红
    "blue":  ( 67, 142, 219),   # 标准蓝色
    "white": (255, 255, 255),   # 白色
    "gray":  (240, 240, 240),   # 浅灰色
}

PRESET_SIZES = {
    # ── 标准冲印尺寸 (@300dpi) ──
    "1inch":      (295, 413),    # 一寸       25×35mm
    "2inch":      (413, 579),    # 二寸       35×49mm
    "small1inch": (260, 378),    # 小一寸     22×32mm
    # ── 各类证件/考试 ──
    "teacher":    (390, 567),    # 教师资格证  33×48mm
    "civil":      (295, 413),    # 国考证      25×35mm (同一寸)
    "ncre":       (144, 192),    # 全国计算机等级考试 (在线报名)
    "student":    (480, 640),    # 大学生图像信息采集 (学信网)
    "gwy":        (295, 413),    # 国家公务员  25×35mm (同一寸)
    "resume":     (295, 413),    # 简历照片    25×35mm (同一寸)
    "passport":   (390, 567),    # 护照        33×48mm
}

# 支持的图片格式
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

# 质量配置
JPEG_QUALITY = 95
PNG_COMPRESSION = 9

# 质量检测阈值
MIN_RESOLUTION = 200
MIN_SHARPNESS = 100.0
MIN_BRIGHTNESS = 50.0
MAX_BRIGHTNESS = 200.0

# 安全限制
MAX_FILE_SIZE_MB = 50  # 最大文件大小
MAX_PIXELS = 50_000_000  # 最大图片像素数 (~7000x7000)

# 全局缓存的人脸检测器（避免重复加载）
_FACE_CASCADE = None


def get_face_cascade():
    """获取缓存的人脸检测器（单例模式）"""
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        _FACE_CASCADE = cv2.CascadeClassifier(cascade_path)
        if _FACE_CASCADE.empty():
            logger.error("无法加载人脸检测器")
            raise RuntimeError("人脸检测器加载失败")
    return _FACE_CASCADE

# ── 参数解析工具 ─────────────────────────────────────────

def parse_bg_color(value: str) -> Tuple[int, int, int]:
    """解析背景颜色，支持预设名称和十六进制"""
    v = value.strip().lower()
    if v in PRESET_COLORS:
        return PRESET_COLORS[v]
    hex_str = v.lstrip("#")
    if len(hex_str) == 6:
        try:
            return (int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16))
        except ValueError:
            pass
    raise ValueError(f"无法识别的颜色: '{value}'  (支持: {', '.join(PRESET_COLORS)} / #RRGGBB)")


def parse_size(value: str) -> Tuple[int, int]:
    """解析输出尺寸，支持预设名称和 WxH 格式"""
    v = value.strip().lower()
    if v in PRESET_SIZES:
        return PRESET_SIZES[v]
    if "x" in v:
        parts = v.split("x")
        if len(parts) == 2:
            try:
                w, h = int(parts[0]), int(parts[1])
                if w > 0 and h > 0:
                    return (w, h)
            except ValueError:
                pass
    raise ValueError(f"无法识别的尺寸: '{value}'  (支持: {', '.join(PRESET_SIZES)} / WxH)")


def validate_image_file(path: Path) -> bool:
    """验证是否为支持的图片文件"""
    return path.is_file() and path.suffix.lower() in SUPPORTED_FORMATS

# ── 人脸检测 ─────────────────────────────────────────────

def detect_face_for_crop(img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    检测人脸位置，返回最大人脸矩形 (x, y, w, h) 或 None
    
    优化点：
    - 使用更严格的参数减少误检
    - 添加最小尺寸限制
    - 使用全局缓存的分类器（提升批量处理性能）
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 使用缓存的分类器
        face_cascade = get_face_cascade()
        
        # 优化参数：scaleFactor=1.1, minNeighbors=5, minSize=(50,50)
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            logger.warning("未检测到人脸")
            return None
            
        # 返回最大的人脸（通常是主要人物）
        main_face = max(faces, key=lambda f: f[2] * f[3])
        logger.info(f"检测到人脸: 位置({main_face[0]}, {main_face[1]}), 大小{main_face[2]}x{main_face[3]}")
        return tuple(main_face)
        
    except Exception as e:
        logger.error(f"人脸检测失败: {e}")
        return None

# ── 智能裁剪 ─────────────────────────────────────────────

def smart_crop_to_chest(img, face_rect=None, target_ratio=3/4):
    """
    智能裁剪到胸部位置
    target_ratio: 宽高比，默认3:4（标准证件照比例）
    """
    height, width = img.shape[:2]

    if face_rect is None:
        crop_height = int(height * 0.7)
        crop_width = int(crop_height * target_ratio)
        x_start = max(0, (width - crop_width) // 2)
        x_end = min(width, x_start + crop_width)
        return img[0:min(height, crop_height), x_start:x_end]

    (fx, fy, fw, fh) = face_rect
    face_center_x = fx + fw // 2

    # 证件照标准：头顶到胸部约为人脸高度的 3.0 倍
    crop_height = int(fh * 3.0)
    crop_width = int(crop_height * target_ratio)

    # 横向：人脸居中
    x_start = max(0, face_center_x - crop_width // 2)
    x_end = min(width, x_start + crop_width)
    if x_end - x_start < crop_width:
        x_start = max(0, x_end - crop_width)

    # 纵向：头顶上方留 0.4 倍人脸高度
    y_start = max(0, fy - int(fh * 0.4))
    y_end = min(height, y_start + crop_height)
    if y_end - y_start < crop_height:
        y_start = max(0, y_end - crop_height)

    return img[y_start:y_end, x_start:x_end]

# ── 边缘羽化 ─────────────────────────────────────────────

def refine_mask(alpha: np.ndarray) -> np.ndarray:
    """
    多步骤边缘优化，消除抠图硬边和颜色溢出：
    1. 闭运算 — 填充小孔洞
    2. 腐蚀   — 去除边缘残留的背景色像素
    3. 高斯模糊 — 柔化边缘过渡（羽化）
    """
    # 闭运算填补小孔洞
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    # 轻微腐蚀，去除紧贴人物边缘的背景色像素
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    alpha = cv2.erode(alpha, kernel_erode, iterations=1)

    # 高斯模糊做边缘羽化
    alpha = cv2.GaussianBlur(alpha, (7, 7), 0)

    return alpha

# ── 图片质量检测 ─────────────────────────────────────────

def check_image_quality(img: np.ndarray) -> Tuple[bool, str]:
    """
    检测图片质量是否符合证件照要求
    返回: (是否合格, 提示信息)
    """
    height, width = img.shape[:2]
    
    # 1. 分辨率检查
    if width < MIN_RESOLUTION or height < MIN_RESOLUTION:
        return False, f"图片分辨率过低 ({width}x{height})，建议至少 600x800"
    
    # 2. 模糊检测（使用拉普拉斯方差）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if laplacian_var < MIN_SHARPNESS:
        return False, f"图片模糊度过高 (清晰度: {laplacian_var:.1f})，建议使用更清晰的照片"
    
    # 3. 亮度检查
    brightness = np.mean(gray)
    if brightness < MIN_BRIGHTNESS:
        return False, f"图片过暗 (亮度: {brightness:.1f})，建议在光线充足环境拍摄"
    elif brightness > MAX_BRIGHTNESS:
        return False, f"图片过亮 (亮度: {brightness:.1f})，建议避免过度曝光"
    
    return True, f"图片质量良好 (分辨率: {width}x{height}, 清晰度: {laplacian_var:.1f}, 亮度: {brightness:.1f})"


# ── 核心生成逻辑 ─────────────────────────────────────────

def generate_id_photo(
    input_path: str, 
    output_path: str, 
    bg_color: Tuple[int, int, int] = (238, 28, 37), 
    output_size: Tuple[int, int] = (413, 579),
    skip_quality_check: bool = False,
    verbose: bool = True
) -> Optional[np.ndarray]:
    """
    使用 rembg 生成证件照
    
    参数:
        input_path: 输入图片路径
        output_path: 输出图片路径
        bg_color: RGB 背景颜色元组
        output_size: (宽, 高) 输出尺寸
        skip_quality_check: 是否跳过质量检查
        verbose: 是否打印 CLI 输出信息
    
    返回:
        成功返回处理后的图片数组，失败返回 None
    """
    if not REMBG_AVAILABLE:
        logger.error("rembg 未安装，无法继续")
        print("❌ rembg 未安装，无法继续。请执行:")
        print("   pip3 install 'rembg[cpu]'")
        return None

    try:
        if verbose:
            print(f"\n{'=' * 56}")
            print("  📸 高级证件照生成器")
            print(f"{'=' * 56}")
            print(f"  输入: {input_path}")
            print(f"  背景: RGB{bg_color}")
            print(f"  尺寸: {output_size[0]}×{output_size[1]}")
            print(f"{'=' * 56}\n")

        # ── 0. 输入验证 ──
        input_path_obj = Path(input_path)
        if not input_path_obj.exists():
            if verbose: print(f"  ❌ 文件不存在: {input_path}")
            logger.error(f"文件不存在: {input_path}")
            return None
            
        file_size = input_path_obj.stat().st_size
        if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
            if verbose: print(f"  ❌ 文件过大 (最大允许 {MAX_FILE_SIZE_MB}MB)")
            logger.error(f"文件过大: {file_size / 1024 / 1024:.1f}MB")
            return None

        # ── 1. 读取图片 ──
        if verbose: print("[1/6] 读取图片...")
        try:
            input_img = Image.open(input_path)
            
            # 安全检查：像素数量
            total_pixels = input_img.size[0] * input_img.size[1]
            if total_pixels > MAX_PIXELS:
                if verbose: print(f"  ❌ 图片像素数过多 ({input_img.size[0]}x{input_img.size[1]})")
                logger.error(f"图片像素数过多: {total_pixels:,}")
                return None
                
            input_img = ImageOps.exif_transpose(input_img)  # 自动修正方向
            img_cv = cv2.cvtColor(np.array(input_img), cv2.COLOR_RGB2BGR)
            if verbose: print(f"  ✅ 原始尺寸: {input_img.size[0]}×{input_img.size[1]}")
        except Exception as e:
            logger.error(f"读取图片失败: {e}")
            if verbose: print(f"  ❌ 读取图片失败: {e}")
            return None

        # ── 2. 质量检测 ──
        if not skip_quality_check:
            if verbose: print("\n[2/6] 检测图片质量...")
            quality_ok, quality_msg = check_image_quality(img_cv)
            if verbose: print(f"  {'✅' if quality_ok else '⚠️ '} {quality_msg}")
            if not quality_ok:
                if verbose: print("  💡 提示: 使用 --skip-quality-check 跳过质量检查")

        # ── 3. 检测人脸 ──
        if verbose: print(f"\n[{'3' if not skip_quality_check else '2'}/6] 检测人脸位置...")
        face_rect = detect_face_for_crop(img_cv)
        if face_rect is not None:
            fx, fy, fw, fh = face_rect
            if verbose: print(f"  ✅ 人脸: ({fx}, {fy}) 大小 {fw}×{fh}")
        else:
            if verbose: print("  ⚠️  未检测到人脸，使用默认裁剪")
            logger.warning("未检测到人脸，使用默认裁剪策略")

        # ── 4. AI 去背景 ──
        if verbose: print(f"\n[{'4' if not skip_quality_check else '3'}/6] AI 去除背景...")
        try:
            session = get_rembg_session()
            output_img = remove(input_img, session=session) if session else remove(input_img)
            img_array = np.array(output_img)  # RGBA, RGB 通道顺序
            del output_img  # 释放内存
            
            # 校验尺寸一致性
            if img_array.shape[:2] != img_cv.shape[:2]:
                logger.warning(f"抠图后尺寸 ({img_array.shape[1]}x{img_array.shape[0]}) 与输入不一致，重置人脸坐标")
                face_rect = None
                
            if verbose: print("  ✅ 背景已去除")
        except Exception as e:
            logger.error(f"AI 去背景失败: {e}")
            if verbose: print(f"  ❌ AI 去背景失败: {e}")
            return None

        # ── 5. 智能裁剪 ──
        if verbose: print(f"\n[{'5' if not skip_quality_check else '4'}/6] 智能裁剪到胸部...")
        try:
            img_cropped = smart_crop_to_chest(img_array, face_rect)
            del img_array # 释放内存
            if verbose: print(f"  ✅ 裁剪: {img_cropped.shape[1]}×{img_cropped.shape[0]}")
        except Exception as e:
            logger.error(f"智能裁剪失败: {e}")
            if verbose: print(f"  ❌ 智能裁剪失败: {e}")
            return None

        # ── 6. 边缘优化 + 合成背景 ──
        if verbose: print(f"\n[{'6' if not skip_quality_check else '5'}/6] 边缘优化 & 合成背景...")
        try:
            height, width = img_cropped.shape[:2]

            # 创建纯色背景 (RGB 顺序)
            bg = np.full((height, width, 3), bg_color, dtype=np.uint8)

            if img_cropped.shape[2] == 4:
                # 提取并优化 alpha 通道
                raw_alpha = img_cropped[:, :, 3]
                smooth_alpha = refine_mask(raw_alpha)

                # Alpha 混合
                alpha_f = smooth_alpha[:, :, np.newaxis].astype(np.float32) / 255.0
                fg = img_cropped[:, :, :3].astype(np.float32)
                result = (fg * alpha_f + bg.astype(np.float32) * (1.0 - alpha_f))
                result = np.clip(result, 0, 255).astype(np.uint8)
                del fg, alpha_f, bg, smooth_alpha, raw_alpha # 释放内存
            else:
                result = img_cropped[:, :, :3]

            # RGB → BGR 并缩放到目标尺寸
            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            out_w, out_h = output_size
            result_bgr = cv2.resize(result_bgr, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

            # 保存（根据扩展名选择格式）
            output_ext = Path(output_path).suffix.lower()
            if output_ext in ['.jpg', '.jpeg']:
                cv2.imwrite(str(output_path), result_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            elif output_ext == '.png':
                cv2.imwrite(str(output_path), result_bgr, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
            else:
                cv2.imwrite(str(output_path), result_bgr)

            if verbose:
                print(f"  ✅ 边缘已优化（闭运算 + 腐蚀 + 羽化）")
                print(f"\n{'=' * 56}")
                print(f"  ✅ 证件照已保存: {output_path}")
                print(f"  ✅ 最终尺寸: {out_w}×{out_h}")
                print(f"{'=' * 56}\n")
            
            logger.info(f"成功生成证件照: {output_path}")
            return result_bgr
            
        except Exception as e:
            logger.error(f"图片合成失败: {e}")
            if verbose: print(f"  ❌ 图片合成失败: {e}")
            return None
            
    except Exception as e:
        logger.error(f"处理过程出现未知错误: {e}")
        if verbose: print(f"❌ 处理失败: {e}")
        return None


# ── 批量处理 ─────────────────────────────────────────────

def batch_process(
    input_dir: Path,
    output_dir: Optional[Path],
    bg_color: Tuple[int, int, int],
    output_size: Tuple[int, int],
    max_workers: int = 4
) -> Tuple[int, int]:
    """
    批量处理文件夹中的所有图片
    
    返回: (成功数量, 失败数量)
    """
    # 查找所有支持的图片文件
    image_files = [f for f in input_dir.iterdir() if validate_image_file(f)]
    
    if not image_files:
        print(f"❌ 在 {input_dir} 中未找到支持的图片文件")
        return 0, 0
    
    # 确定输出目录
    if output_dir is None:
        output_dir = input_dir / "id_photos"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'=' * 56}")
    print(f"  📦 批量处理模式")
    print(f"{'=' * 56}")
    print(f"  输入目录: {input_dir}")
    print(f"  输出目录: {output_dir}")
    print(f"  图片数量: {len(image_files)}")
    print(f"  并发数: {max_workers}")
    print(f"{'=' * 56}\n")
    
    lock = threading.Lock()
    success_count = 0
    fail_count = 0
    
    def process_single(img_path: Path) -> bool:
        """处理单张图片"""
        try:
            bg_name = f"rgb{''.join(str(c) for c in bg_color)}"
            output_path = output_dir / f"{img_path.stem}_id_{bg_name}{img_path.suffix}"
            result = generate_id_photo(
                str(img_path), 
                str(output_path), 
                bg_color, 
                output_size,
                skip_quality_check=True,  # 批量模式跳过质量检查以提高速度
                verbose=False             # 批量模式关闭单张照片的详细打印
            )
            return result is not None
        except Exception as e:
            logger.error(f"处理 {img_path.name} 失败: {e}")
            return False
    
    # 使用线程池并发处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single, img): img for img in image_files}
        
        for future in as_completed(futures):
            img_path = futures[future]
            try:
                if future.result():
                    with lock:
                        success_count += 1
                        current_success = success_count
                        current_fail = fail_count
                    print(f"✅ [{current_success + current_fail}/{len(image_files)}] {img_path.name}")
                else:
                    with lock:
                        fail_count += 1
                        current_success = success_count
                        current_fail = fail_count
                    print(f"❌ [{current_success + current_fail}/{len(image_files)}] {img_path.name}")
            except Exception as e:
                with lock:
                    fail_count += 1
                    current_success = success_count
                    current_fail = fail_count
                logger.error(f"处理 {img_path.name} 时发生异常: {e}")
                print(f"❌ [{current_success + current_fail}/{len(image_files)}] {img_path.name} - {e}")
    
    print(f"\n{'=' * 56}")
    print(f"  批量处理完成")
    print(f"  成功: {success_count} | 失败: {fail_count}")
    print(f"{'=' * 56}\n")
    
    return success_count, fail_count

# ── CLI ──────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description="高级证件照生成器 — AI抠图 + 智能裁剪 + 边缘羽化 + 批量处理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""\
背景颜色: {', '.join(PRESET_COLORS)} 或 #RRGGBB
输出尺寸: {', '.join(f'{k}({v[0]}x{v[1]})' for k, v in list(PRESET_SIZES.items())[:5])} 等，或 WxH

示例:
  # 单张处理
  %(prog)s -i photo.jpg                         # 红底 二寸
  %(prog)s -i photo.jpg -bg blue                # 蓝底
  %(prog)s -i photo.jpg -bg white -s 1inch      # 白底 一寸
  %(prog)s -i photo.jpg -bg "#0066CC" -s 295x413
  
  # 批量处理
  %(prog)s -i photos/ --batch                   # 批量处理文件夹
  %(prog)s -i photos/ --batch -o output/ -bg blue --workers 8
""",
    )
    p.add_argument("-i", "--input", required=True, help="输入图片路径或文件夹（批量模式）")
    p.add_argument("-o", "--output", default=None, help="输出路径（默认自动生成）")
    p.add_argument("-bg", "--background", default="red",
                   help="背景颜色: red / blue / white / gray / #RRGGBB（默认 red）")
    p.add_argument("-s", "--size", default="2inch",
                   help="输出尺寸: 1inch / 2inch / passport / resume 等，或 WxH（默认 2inch）")
    p.add_argument("--batch", action="store_true", help="批量处理模式（输入为文件夹）")
    p.add_argument("--workers", type=int, default=4, help="批量处理并发数（默认 4）")
    p.add_argument("--skip-quality-check", action="store_true", help="跳过图片质量检查")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ 路径不存在: {input_path}")
        sys.exit(1)

    # 解析背景颜色
    try:
        bg_color = parse_bg_color(args.background)
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)

    # 解析输出尺寸
    try:
        output_size = parse_size(args.size)
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)

    # 批量处理模式
    if args.batch:
        if not input_path.is_dir():
            print(f"❌ 批量模式需要输入文件夹路径，但得到: {input_path}")
            sys.exit(1)
        
        output_dir = Path(args.output) if args.output else None
        success, fail = batch_process(
            input_path, 
            output_dir, 
            bg_color, 
            output_size,
            max_workers=args.workers
        )
        sys.exit(0 if fail == 0 else 1)
    
    # 单张处理模式
    if not validate_image_file(input_path):
        print(f"❌ 不支持的文件格式: {input_path.suffix}")
        print(f"   支持的格式: {', '.join(SUPPORTED_FORMATS)}")
        sys.exit(1)

    # 确定输出路径
    if args.output:
        output_path = args.output
    else:
        bg_name = args.background.lstrip("#").lower()
        output_path = f"{input_path.stem}_id_{bg_name}.jpg"

    # 生成证件照
    result = generate_id_photo(
        str(input_path), 
        output_path, 
        bg_color, 
        output_size,
        skip_quality_check=args.skip_quality_check
    )
    
    sys.exit(0 if result is not None else 1)


if __name__ == "__main__":
    main()
