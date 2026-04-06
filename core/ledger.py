"""
哈希链存证模块 —— 模拟区块链的不可篡改记录

原理：
每条记录包含前一条记录的哈希，形成链式结构。
任何篡改都会导致后续所有记录的哈希校验失败。
"""
import hashlib
import json
import os
import time
from typing import List, Optional

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LEDGER_PATH


def _compute_block_hash(block: dict) -> str:
    """计算区块哈希"""
    content = json.dumps({
        "index": block["index"],
        "timestamp": block["timestamp"],
        "data": block["data"],
        "prev_hash": block["prev_hash"],
    }, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()


def _load_ledger() -> List[dict]:
    """加载账本"""
    if os.path.exists(LEDGER_PATH):
        with open(LEDGER_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_ledger(chain: List[dict]):
    """保存账本"""
    with open(LEDGER_PATH, "w", encoding="utf-8") as f:
        json.dump(chain, f, ensure_ascii=False, indent=2)


def add_record(data: dict) -> dict:
    """
    添加一条存证记录

    Args:
        data: 要存证的数据（如模型确权信息）

    Returns:
        新添加的区块
    """
    chain = _load_ledger()

    # 创建新区块
    block = {
        "index": len(chain),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data": data,
        "prev_hash": chain[-1]["hash"] if chain else "0" * 64,
    }
    block["hash"] = _compute_block_hash(block)

    chain.append(block)
    _save_ledger(chain)

    return block


def verify_chain() -> tuple:
    """
    验证整条哈希链的完整性

    Returns:
        (是否完整, 错误信息)
    """
    chain = _load_ledger()

    if not chain:
        return True, "账本为空"

    for i, block in enumerate(chain):
        # 验证哈希
        expected_hash = _compute_block_hash(block)
        if block["hash"] != expected_hash:
            return False, f"区块 {i} 哈希不匹配：已被篡改"

        # 验证链接
        if i > 0 and block["prev_hash"] != chain[i - 1]["hash"]:
            return False, f"区块 {i} 的前向链接断裂：已被篡改"

    return True, f"账本完整，共 {len(chain)} 条记录"


def get_all_records() -> List[dict]:
    """获取所有存证记录"""
    return _load_ledger()


def get_record_by_index(index: int) -> Optional[dict]:
    """按索引获取记录"""
    chain = _load_ledger()
    if 0 <= index < len(chain):
        return chain[index]
    return None


def search_records(owner_id: str = None, model_name: str = None) -> List[dict]:
    """搜索存证记录"""
    chain = _load_ledger()
    results = []
    for block in chain:
        data = block.get("data", {})
        if owner_id and data.get("owner_id") != owner_id:
            continue
        if model_name and data.get("model_name") != model_name:
            continue
        results.append(block)
    return results
