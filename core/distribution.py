"""
分发溯源模块 —— 每份分发文件嵌入唯一指纹，泄露后追溯来源

流程：
1. 内容方上传原始文件
2. 每次分发给不同平台/用户时，嵌入唯一指纹
3. 记录分发信息（平台、IP、时间、指纹）到存证链
4. 发现泄露后，提取指纹 → 比对分发记录 → 定位泄露源头
"""
import os
import json
import time
import hashlib
from typing import List, Optional

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR
from core.ledger import add_record, get_all_records

DISTRIBUTION_LOG = os.path.join(DATA_DIR, "distributions.json")


def _load_distributions() -> List[dict]:
    if os.path.exists(DISTRIBUTION_LOG):
        with open(DISTRIBUTION_LOG, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_distributions(records: List[dict]):
    with open(DISTRIBUTION_LOG, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def register_distribution(
    file_name: str,
    file_hash: str,
    platform: str,
    ip_address: str,
    user_id: str,
    fingerprint: str,
) -> dict:
    """
    登记一次分发记录

    Args:
        file_name: 文件名
        file_hash: 原始文件哈希
        platform: 分发平台（如 抖音、B站、YouTube）
        ip_address: 接收方IP
        user_id: 接收方用户ID
        fingerprint: 嵌入的唯一指纹

    Returns:
        分发记录
    """
    record = {
        "id": hashlib.sha256(f"{fingerprint}-{time.time()}".encode()).hexdigest()[:12],
        "file_name": file_name,
        "file_hash": file_hash,
        "platform": platform,
        "ip_address": ip_address,
        "user_id": user_id,
        "fingerprint": fingerprint,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "active",
    }

    # 保存到分发日志
    distributions = _load_distributions()
    distributions.append(record)
    _save_distributions(distributions)

    # 同时上链存证
    add_record({
        "type": "distribution",
        "distribution_id": record["id"],
        "file_name": file_name,
        "platform": platform,
        "user_id": user_id,
        "fingerprint_hash": hashlib.sha256(fingerprint.encode()).hexdigest()[:16],
    })

    return record


def trace_leak(extracted_fingerprint: str) -> dict:
    """
    根据提取的指纹追溯泄露来源

    Args:
        extracted_fingerprint: 从泄露文件中提取的指纹

    Returns:
        溯源报告
    """
    distributions = _load_distributions()

    if not distributions:
        return {
            "found": False,
            "message": "无分发记录",
            "matches": [],
        }

    # 精确匹配
    exact_match = None
    for d in distributions:
        if d["fingerprint"] == extracted_fingerprint:
            exact_match = d
            break

    # 模糊匹配（指纹可能在传输中有损）
    fuzzy_matches = []
    for d in distributions:
        fp = d["fingerprint"]
        # 字符级相似度
        common_len = min(len(extracted_fingerprint), len(fp))
        if common_len == 0:
            continue
        matches = sum(1 for a, b in zip(extracted_fingerprint[:common_len], fp[:common_len]) if a == b)
        similarity = matches / max(len(extracted_fingerprint), len(fp))

        if similarity > 0.5:
            fuzzy_matches.append({
                **d,
                "similarity": round(similarity, 4),
            })

    fuzzy_matches.sort(key=lambda x: x["similarity"], reverse=True)

    report = {
        "found": exact_match is not None or len(fuzzy_matches) > 0,
        "exact_match": exact_match,
        "fuzzy_matches": fuzzy_matches[:5],
        "trace_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "extracted_fingerprint": extracted_fingerprint,
    }

    if exact_match:
        report["conclusion"] = (
            f"泄露源确认：平台「{exact_match['platform']}」，"
            f"用户「{exact_match['user_id']}」，"
            f"IP「{exact_match['ip_address']}」，"
            f"分发时间「{exact_match['timestamp']}」"
        )
    elif fuzzy_matches:
        top = fuzzy_matches[0]
        report["conclusion"] = (
            f"最可能的泄露源：平台「{top['platform']}」，"
            f"用户「{top['user_id']}」，"
            f"IP「{top['ip_address']}」，"
            f"相似度 {top['similarity']*100:.1f}%"
        )
    else:
        report["conclusion"] = "未找到匹配的分发记录"

    # 溯源结果上链
    add_record({
        "type": "trace",
        "found": report["found"],
        "conclusion": report.get("conclusion", ""),
        "fingerprint_preview": extracted_fingerprint[:20],
    })

    return report


def get_all_distributions() -> List[dict]:
    """获取所有分发记录"""
    return _load_distributions()


def get_distribution_stats() -> dict:
    """获取分发统计"""
    distributions = _load_distributions()
    platforms = {}
    for d in distributions:
        p = d["platform"]
        platforms[p] = platforms.get(p, 0) + 1

    return {
        "total": len(distributions),
        "platforms": platforms,
    }
