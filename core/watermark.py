"""
模型水印模块 —— 权重级无损水印嵌入与提取

原理：
1. 选取模型中特定层的权重参数
2. 根据密钥生成伪随机位置序列
3. 在这些位置上叠加微小扰动来编码水印信息
4. 提取时用相同密钥定位并解码
"""
import hashlib
import numpy as np
import torch
import copy
from typing import Tuple, Optional

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import WATERMARK_STRENGTH, WATERMARK_KEY_LENGTH


def _get_watermark_positions(
    param_size: int, key: str, num_bits: int
) -> np.ndarray:
    """根据密钥生成水印嵌入位置（伪随机、可复现）"""
    seed = int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.RandomState(seed)
    positions = rng.choice(param_size, size=num_bits, replace=False)
    return positions


def _select_target_layer(model: torch.nn.Module) -> Tuple[str, torch.nn.Parameter]:
    """选取最适合嵌入水印的层（选参数量最大的权重层）"""
    best_name, best_param = None, None
    best_size = 0
    for name, param in model.named_parameters():
        if "weight" in name and param.numel() > best_size:
            best_name = name
            best_param = param
            best_size = param.numel()
    if best_param is None:
        raise ValueError("模型中没有找到可用的权重层")
    return best_name, best_param


def embed_watermark(
    model: torch.nn.Module,
    owner_id: str,
    secret_key: str,
    strength: float = WATERMARK_STRENGTH,
    num_bits: int = WATERMARK_KEY_LENGTH,
) -> Tuple[torch.nn.Module, dict]:
    """
    在模型权重中嵌入水印

    Args:
        model: PyTorch 模型
        owner_id: 所有者标识（会被编码为水印信息）
        secret_key: 水印密钥（用于确定嵌入位置）
        strength: 嵌入强度
        num_bits: 水印位数

    Returns:
        (加水印的模型, 水印元数据)
    """
    watermarked_model = copy.deepcopy(model)

    # 将 owner_id 哈希为固定长度的二进制水印
    owner_hash = hashlib.sha256(owner_id.encode()).hexdigest()
    watermark_bits = bin(int(owner_hash[:num_bits // 4], 16))[2:].zfill(num_bits)
    watermark_bits = [int(b) for b in watermark_bits[:num_bits]]

    # 选取目标层
    layer_name, _ = _select_target_layer(watermarked_model)

    # 在加水印模型中找到对应参数
    target_param = None
    for name, param in watermarked_model.named_parameters():
        if name == layer_name:
            target_param = param
            break

    # 获取嵌入位置
    flat_data = target_param.data.view(-1)
    positions = _get_watermark_positions(flat_data.numel(), secret_key, num_bits)

    # 嵌入水印：bit=1 加扰动，bit=0 减扰动
    with torch.no_grad():
        for i, pos in enumerate(positions):
            delta = strength * (1 if watermark_bits[i] else -1)
            flat_data[pos] += delta

    metadata = {
        "layer_name": layer_name,
        "num_bits": num_bits,
        "strength": strength,
        "owner_id": owner_id,
        "watermark_hash": owner_hash[:16],
    }

    return watermarked_model, metadata


def extract_watermark(
    original_model: torch.nn.Module,
    watermarked_model: torch.nn.Module,
    secret_key: str,
    num_bits: int = WATERMARK_KEY_LENGTH,
    strength: float = WATERMARK_STRENGTH,
) -> Tuple[list, float]:
    """
    从模型中提取水印

    Args:
        original_model: 原始模型（用于对比）
        watermarked_model: 疑似含水印的模型
        secret_key: 水印密钥
        num_bits: 水印位数
        strength: 嵌入强度

    Returns:
        (提取的水印bits, 置信度)
    """
    layer_name, _ = _select_target_layer(original_model)

    # 获取两个模型对应层的参数
    orig_param = None
    wm_param = None
    for name, param in original_model.named_parameters():
        if name == layer_name:
            orig_param = param
            break
    for name, param in watermarked_model.named_parameters():
        if name == layer_name:
            wm_param = param
            break

    orig_flat = orig_param.data.view(-1)
    wm_flat = wm_param.data.view(-1)
    positions = _get_watermark_positions(orig_flat.numel(), secret_key, num_bits)

    # 提取水印
    extracted_bits = []
    for pos in positions:
        diff = (wm_flat[pos] - orig_flat[pos]).item()
        extracted_bits.append(1 if diff > 0 else 0)

    # 计算置信度（正确提取比例的统计显著性）
    confidence = sum(1 for b in extracted_bits if b == extracted_bits[0]) / len(extracted_bits)

    return extracted_bits, confidence


def verify_ownership(
    original_model: torch.nn.Module,
    suspect_model: torch.nn.Module,
    owner_id: str,
    secret_key: str,
    num_bits: int = WATERMARK_KEY_LENGTH,
    strength: float = WATERMARK_STRENGTH,
    threshold: float = 0.85,
) -> Tuple[bool, float, dict]:
    """
    验证模型归属权

    Args:
        original_model: 原始模型
        suspect_model: 待验证模型
        owner_id: 声称的所有者ID
        secret_key: 水印密钥
        threshold: 匹配阈值

    Returns:
        (是否匹配, 匹配率, 详细信息)
    """
    # 提取水印
    extracted_bits, confidence = extract_watermark(
        original_model, suspect_model, secret_key, num_bits, strength
    )

    # 重建期望的水印
    owner_hash = hashlib.sha256(owner_id.encode()).hexdigest()
    expected_bits = bin(int(owner_hash[:num_bits // 4], 16))[2:].zfill(num_bits)
    expected_bits = [int(b) for b in expected_bits[:num_bits]]

    # 计算匹配率
    matches = sum(1 for a, b in zip(extracted_bits, expected_bits) if a == b)
    match_rate = matches / num_bits

    is_owner = match_rate >= threshold

    details = {
        "match_rate": match_rate,
        "threshold": threshold,
        "num_bits": num_bits,
        "matched_bits": matches,
        "is_owner": is_owner,
    }

    return is_owner, match_rate, details
