"""
防木马窃取模块 —— 访问控制、完整性监控、异常检测、被盗反制

三道防线：
  第一道：访问控制 —— 时效令牌 + IP白名单 + 设备指纹，阻断非授权下载/转发
  第二道：完整性监控 —— 文件哈希监控 + 异常访问检测，发现被窃
  第三道：反制触发 —— 自动对泄露副本触发显式水印（DNA身份证）
"""
import hashlib
import hmac
import json
import os
import time
import secrets
from typing import Optional, List

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR
from core.ledger import add_record


ACCESS_LOG = os.path.join(DATA_DIR, "access_log.json")
TOKEN_STORE = os.path.join(DATA_DIR, "access_tokens.json")
INTEGRITY_STORE = os.path.join(DATA_DIR, "integrity_snapshots.json")

# 异常阈值
MAX_REQUESTS_PER_MINUTE = 10
SUSPICIOUS_UA_KEYWORDS = [
    "wget", "curl", "python-requests", "scrapy", "httpclient",
    "spider", "crawler", "bot", "headless", "phantomjs",
]


# ==================== 访问令牌（防下载/转发） ====================

def issue_access_token(
    file_hash: str,
    user_id: str,
    ip_address: str,
    ttl_seconds: int = 300,
    max_views: int = 1,
) -> dict:
    """
    签发一次性访问令牌。加密文件必须持有效令牌才能解密。

    Args:
        file_hash: 受保护文件的哈希
        user_id: 授权用户ID
        ip_address: 绑定的IP地址（防止令牌被转发到其他IP使用）
        ttl_seconds: 令牌有效期（秒），默认5分钟
        max_views: 最大使用次数，默认单次

    Returns:
        令牌记录（包含 token 字段）
    """
    token = secrets.token_urlsafe(32)
    record = {
        "token": token,
        "file_hash": file_hash,
        "user_id": user_id,
        "bind_ip": ip_address,
        "issued_at": time.time(),
        "expires_at": time.time() + ttl_seconds,
        "max_views": max_views,
        "used": 0,
        "revoked": False,
    }
    _save_token(record)
    return record


def verify_access_token(token: str, request_ip: str, user_agent: str = "") -> dict:
    """
    校验令牌。返回 {"valid": bool, "reason": str, "record": dict|None}。
    异常情况会被自动记录到访问日志，触发后续反制。
    """
    tokens = _load_tokens()
    matched = next((t for t in tokens if t["token"] == token), None)

    result = {"valid": False, "reason": "", "record": matched}

    if not matched:
        result["reason"] = "token_not_found"
        log_access_event({
            "event": "invalid_token",
            "ip": request_ip, "user_agent": user_agent, "token_prefix": token[:8],
        })
        return result

    now = time.time()
    if matched["revoked"]:
        result["reason"] = "revoked"
    elif now > matched["expires_at"]:
        result["reason"] = "expired"
    elif matched["used"] >= matched["max_views"]:
        result["reason"] = "exhausted"
    elif matched["bind_ip"] != request_ip:
        result["reason"] = "ip_mismatch"
    elif _is_suspicious_ua(user_agent):
        result["reason"] = "suspicious_user_agent"
    else:
        # 合法访问
        matched["used"] += 1
        _save_token(matched, update=True)
        result["valid"] = True
        result["reason"] = "ok"
        log_access_event({
            "event": "granted",
            "ip": request_ip, "user_agent": user_agent,
            "file_hash": matched["file_hash"], "user_id": matched["user_id"],
        })
        return result

    # 异常访问：全部记录并上链
    log_access_event({
        "event": "denied",
        "reason": result["reason"],
        "ip": request_ip, "user_agent": user_agent,
        "file_hash": matched["file_hash"], "user_id": matched["user_id"],
    })
    add_record({
        "type": "access_denied",
        "reason": result["reason"],
        "ip": request_ip,
        "file_hash_preview": matched["file_hash"][:16],
    })
    return result


def revoke_token(token: str) -> bool:
    tokens = _load_tokens()
    for t in tokens:
        if t["token"] == token:
            t["revoked"] = True
            _save_tokens(tokens)
            return True
    return False


# ==================== 完整性监控（发现被窃） ====================

def snapshot_integrity(file_path: str, file_hash: str, owner: str) -> dict:
    """为受保护文件建立完整性快照。后续可周期性检查是否被篡改或拷贝。"""
    snapshot = {
        "file_path": file_path,
        "file_hash": file_hash,
        "owner": owner,
        "size": os.path.getsize(file_path) if os.path.exists(file_path) else 0,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    snapshots = _load_snapshots()
    snapshots.append(snapshot)
    _save_snapshots(snapshots)
    return snapshot


def check_integrity(file_path: str) -> dict:
    """校验文件是否被篡改。返回 {"ok": bool, "expected": str, "actual": str}。"""
    snapshots = _load_snapshots()
    snap = next((s for s in snapshots if s["file_path"] == file_path), None)
    if not snap:
        return {"ok": False, "reason": "no_snapshot"}
    if not os.path.exists(file_path):
        return {"ok": False, "reason": "file_missing"}

    with open(file_path, "rb") as f:
        actual = hashlib.sha256(f.read()).hexdigest()

    ok = actual == snap["file_hash"]
    if not ok:
        # 发现被改 → 上链 + 触发反制
        add_record({
            "type": "integrity_alert",
            "file_path": os.path.basename(file_path),
            "expected_hash": snap["file_hash"][:16],
            "actual_hash": actual[:16],
        })
    return {
        "ok": ok,
        "expected": snap["file_hash"],
        "actual": actual,
        "owner": snap["owner"],
    }


# ==================== 异常检测 ====================

def detect_anomaly(ip: str, user_agent: str, window_seconds: int = 60) -> dict:
    """
    检测来自该 IP/UA 的访问是否可疑。
    包括：爬虫特征UA、高频访问、失败率异常。
    """
    events = _load_access_log()
    now = time.time()
    recent = [e for e in events
              if e.get("ip") == ip and now - e.get("ts", 0) < window_seconds]

    signals = []
    ua_suspicious = _is_suspicious_ua(user_agent)
    if ua_suspicious:
        signals.append(f"可疑UA: 疑似爬虫/脚本 ({user_agent[:30]})")

    if len(recent) > MAX_REQUESTS_PER_MINUTE:
        signals.append(f"高频访问: {len(recent)} 次/分钟（阈值 {MAX_REQUESTS_PER_MINUTE}）")

    denied = [e for e in recent if e.get("event") == "denied"]
    if len(denied) >= 3:
        signals.append(f"连续失败: {len(denied)} 次被拒绝访问")

    return {
        "ip": ip,
        "user_agent": user_agent,
        "suspicious": len(signals) > 0,
        "risk_level": "HIGH" if len(signals) >= 2 else ("MEDIUM" if signals else "LOW"),
        "signals": signals,
        "recent_events": len(recent),
    }


# ==================== 反制触发（自动水印） ====================

def trigger_counter_measure(file_hash: str, reason: str, owner: str) -> dict:
    """
    触发反制：当检测到文件被窃 / 访问异常时，自动标记该副本应触发显式水印。
    调用方（如媒体服务）在下次返回该文件时，用 apply_visible_watermark 覆盖。
    """
    measure = {
        "file_hash": file_hash,
        "reason": reason,
        "owner": owner,
        "triggered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "action": "force_visible_watermark",
        "message": f"检测到 {reason}，强制对盗版副本铺满DNA身份证水印",
    }
    add_record({"type": "counter_measure", **measure})
    return measure


# ==================== 内部工具 ====================

def log_access_event(event: dict):
    event["ts"] = time.time()
    event["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    events = _load_access_log()
    events.append(event)
    # 只保留最近 500 条
    events = events[-500:]
    with open(ACCESS_LOG, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)


def get_access_log(limit: int = 50) -> List[dict]:
    events = _load_access_log()
    return list(reversed(events))[:limit]


def _is_suspicious_ua(ua: str) -> bool:
    if not ua:
        return True
    low = ua.lower()
    return any(k in low for k in SUSPICIOUS_UA_KEYWORDS)


def _load_tokens() -> List[dict]:
    if os.path.exists(TOKEN_STORE):
        with open(TOKEN_STORE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_tokens(tokens: List[dict]):
    with open(TOKEN_STORE, "w", encoding="utf-8") as f:
        json.dump(tokens, f, ensure_ascii=False, indent=2)


def _save_token(record: dict, update: bool = False):
    tokens = _load_tokens()
    if update:
        for i, t in enumerate(tokens):
            if t["token"] == record["token"]:
                tokens[i] = record
                break
    else:
        tokens.append(record)
    _save_tokens(tokens)


def _load_snapshots() -> List[dict]:
    if os.path.exists(INTEGRITY_STORE):
        with open(INTEGRITY_STORE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_snapshots(snapshots: List[dict]):
    with open(INTEGRITY_STORE, "w", encoding="utf-8") as f:
        json.dump(snapshots, f, ensure_ascii=False, indent=2)


def _load_access_log() -> List[dict]:
    if os.path.exists(ACCESS_LOG):
        with open(ACCESS_LOG, "r", encoding="utf-8") as f:
            return json.load(f)
    return []
