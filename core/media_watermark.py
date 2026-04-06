"""
影视文件水印模块 —— 隐式水印 + 显式水印

隐式水印（Invisible）：
  - 基于 DCT（离散余弦变换）将指纹嵌入频域
  - 肉眼不可见，但可提取验证
  - 每次分发嵌入不同指纹，用于溯源

显式水印（Visible）：
  - 被盗触发后，铺满画面的所有权标识
  - 作为影视作品的"DNA身份证"
"""
import cv2
import numpy as np
import hashlib
import time
import os
from typing import Tuple, Optional

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import WATERMARKED_DIR


# ==================== 隐式水印（DCT频域） ====================

def _text_to_bits(text: str) -> list:
    """文本转二进制位序列"""
    bits = []
    for char in text.encode('utf-8'):
        for i in range(7, -1, -1):
            bits.append((char >> i) & 1)
    return bits


def _bits_to_text(bits: list) -> str:
    """二进制位序列转文本"""
    chars = []
    for i in range(0, len(bits) - 7, 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        if byte == 0:
            break
        chars.append(chr(byte))
    return ''.join(chars)


def embed_invisible_watermark(
    image: np.ndarray,
    fingerprint: str,
    strength: float = 25.0,
) -> np.ndarray:
    """
    在图片中嵌入隐式水印（DCT频域）

    Args:
        image: 原始图片 (BGR)
        fingerprint: 要嵌入的指纹信息（如 "platform:douyin|ip:192.168.1.1|time:2024"）
        strength: 嵌入强度

    Returns:
        嵌入水印后的图片
    """
    # 转为 YUV，在 Y 通道嵌入（亮度通道，人眼不敏感）
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV).astype(np.float64)
    y_channel = img_yuv[:, :, 0]

    # 将指纹转为二进制
    bits = _text_to_bits(fingerprint)
    # 补零对齐到 8 的倍数
    while len(bits) % 8 != 0:
        bits.append(0)

    # 在 8x8 块中嵌入
    h, w = y_channel.shape
    bit_idx = 0

    for i in range(0, h - 7, 8):
        for j in range(0, w - 7, 8):
            if bit_idx >= len(bits):
                break

            block = y_channel[i:i+8, j:j+8]
            dct_block = cv2.dct(block)

            # 在中频系数 (4,3) 和 (3,4) 中嵌入
            if bits[bit_idx] == 1:
                dct_block[4, 3] = abs(dct_block[4, 3]) + strength
            else:
                dct_block[4, 3] = -(abs(dct_block[4, 3]) + strength)

            y_channel[i:i+8, j:j+8] = cv2.idct(dct_block)
            bit_idx += 1

        if bit_idx >= len(bits):
            break

    img_yuv[:, :, 0] = y_channel
    result = cv2.cvtColor(img_yuv.astype(np.uint8), cv2.COLOR_YUV2BGR)
    return result


def extract_invisible_watermark(
    watermarked_image: np.ndarray,
    fingerprint_length: int,
    strength: float = 25.0,
) -> str:
    """
    从图片中提取隐式水印

    Args:
        watermarked_image: 含水印图片
        fingerprint_length: 指纹字符数（需要知道长度）
        strength: 嵌入时的强度

    Returns:
        提取的指纹信息
    """
    img_yuv = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2YUV).astype(np.float64)
    y_channel = img_yuv[:, :, 0]

    num_bits = fingerprint_length * 8
    bits = []
    h, w = y_channel.shape

    for i in range(0, h - 7, 8):
        for j in range(0, w - 7, 8):
            if len(bits) >= num_bits:
                break

            block = y_channel[i:i+8, j:j+8]
            dct_block = cv2.dct(block)

            # 提取：正值为1，负值为0
            if dct_block[4, 3] > 0:
                bits.append(1)
            else:
                bits.append(0)

        if len(bits) >= num_bits:
            break

    return _bits_to_text(bits)


# ==================== 显式水印（可见） ====================

def apply_visible_watermark(
    image: np.ndarray,
    owner_text: str,
    opacity: float = 0.3,
    tile: bool = True,
) -> np.ndarray:
    """
    在图片上铺满显式水印（DNA身份证效果）

    Args:
        image: 原始图片
        owner_text: 水印文字（如所有者名称、版权声明）
        opacity: 不透明度 (0-1)
        tile: 是否平铺满整个画面

    Returns:
        带显式水印的图片
    """
    result = image.copy()
    h, w = result.shape[:2]

    # 创建水印层
    overlay = result.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(w, h) / 800)
    thickness = max(1, int(font_scale * 2))
    color = (0, 0, 255)  # 红色

    if tile:
        # 平铺水印 —— 旋转45度铺满
        text_size = cv2.getTextSize(owner_text, font, font_scale, thickness)[0]
        step_x = text_size[0] + 60
        step_y = text_size[1] + 80

        for y in range(-h, h * 2, step_y):
            for x in range(-w, w * 2, step_x):
                # 旋转绘制
                center = (x + text_size[0] // 2, y + text_size[1] // 2)
                M = cv2.getRotationMatrix2D(center, -35, 1.0)

                # 在临时图层上画文字再旋转太复杂，直接斜着放
                pos_x = x + (y % (step_x * 2)) // 3  # 错位排列
                cv2.putText(overlay, owner_text, (pos_x, y),
                           font, font_scale, color, thickness, cv2.LINE_AA)
    else:
        # 单个居中水印
        text_size = cv2.getTextSize(owner_text, font, font_scale * 2, thickness * 2)[0]
        x = (w - text_size[0]) // 2
        y = (h + text_size[1]) // 2
        cv2.putText(overlay, owner_text, (x, y),
                   font, font_scale * 2, color, thickness * 2, cv2.LINE_AA)

    # 混合
    result = cv2.addWeighted(overlay, opacity, result, 1 - opacity, 0)

    # 添加边框警告
    border_text = "UNAUTHORIZED COPY - " + owner_text
    cv2.putText(result, border_text, (10, 30),
               font, font_scale * 0.7, (0, 0, 255), thickness, cv2.LINE_AA)
    cv2.putText(result, border_text, (10, h - 15),
               font, font_scale * 0.7, (0, 0, 255), thickness, cv2.LINE_AA)

    return result


# ==================== 视频处理 ====================

def process_video_watermark(
    input_path: str,
    output_path: str,
    fingerprint: str,
    visible: bool = False,
    owner_text: str = "",
    max_frames: int = 300,  # 限制处理帧数（演示用）
) -> dict:
    """
    给视频嵌入水印

    Args:
        input_path: 输入视频路径
        output_path: 输出视频路径
        fingerprint: 隐式指纹
        visible: 是否同时加显式水印
        owner_text: 显式水印文字
        max_frames: 最大处理帧数

    Returns:
        处理信息
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频：{input_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    processed = 0
    for _ in range(min(total_frames, max_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        # 嵌入隐式水印
        frame = embed_invisible_watermark(frame, fingerprint)

        # 可选：加显式水印
        if visible and owner_text:
            frame = apply_visible_watermark(frame, owner_text)

        out.write(frame)
        processed += 1

    cap.release()
    out.release()

    return {
        "frames_processed": processed,
        "fps": fps,
        "resolution": f"{w}x{h}",
        "fingerprint": fingerprint,
        "visible_watermark": visible,
    }


# ==================== 工具函数 ====================

def generate_fingerprint(platform: str, ip_address: str, user_id: str = "") -> str:
    """生成分发指纹"""
    timestamp = time.strftime("%Y%m%d%H%M%S")
    raw = f"p:{platform}|ip:{ip_address}|u:{user_id}|t:{timestamp}"
    # 限制长度（DCT嵌入容量有限）
    if len(raw) > 60:
        raw = raw[:60]
    return raw


def compare_fingerprints(extracted: str, candidates: list) -> list:
    """
    比对指纹，找到最匹配的分发记录

    Args:
        extracted: 提取的指纹
        candidates: 候选指纹列表 [{"fingerprint": "...", ...}, ...]

    Returns:
        按相似度排序的匹配结果
    """
    results = []
    for candidate in candidates:
        fp = candidate.get("fingerprint", "")
        # 计算字符级相似度
        matches = sum(1 for a, b in zip(extracted, fp) if a == b)
        max_len = max(len(extracted), len(fp), 1)
        similarity = matches / max_len
        results.append({
            **candidate,
            "similarity": round(similarity, 4),
        })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results
