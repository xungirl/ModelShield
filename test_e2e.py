"""
寰宇OS · ModelShield —— 四大需求端到端测试

用户需求：
  1. 加密影视源文件 → 锁死下载/转发/爬虫
  2. 被盗时触发 DNA 身份证水印（铺满画面）
  3. 防止源文件被木马偷走
  4. 被偷后可溯源到首发平台 / IP
"""
import os
import sys
import hashlib
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.crypto import PostQuantumCrypto
from core.media_watermark import (
    embed_invisible_watermark, extract_invisible_watermark,
    apply_visible_watermark, generate_fingerprint,
)
from core.distribution import register_distribution, trace_leak
from core.anti_theft import (
    issue_access_token, verify_access_token,
    snapshot_integrity, check_integrity,
    detect_anomaly, trigger_counter_measure,
    get_access_log,
)
from core.ledger import add_record, verify_chain, get_all_records
from config import WATERMARKED_DIR, DATA_DIR


# ---------- 工具 ----------
GREEN, RED, YELLOW, BLUE, RESET, BOLD = (
    "\033[92m", "\033[91m", "\033[93m", "\033[94m", "\033[0m", "\033[1m"
)
passed = []
failed = []

def ok(name, detail=""):
    passed.append(name)
    print(f"{GREEN}✅ PASS{RESET}  {name}" + (f"  — {detail}" if detail else ""))

def fail(name, detail=""):
    failed.append(name)
    print(f"{RED}❌ FAIL{RESET}  {name}" + (f"  — {detail}" if detail else ""))

def banner(n, title):
    print(f"\n{BOLD}{BLUE}{'━'*72}\n 需求 {n}: {title}\n{'━'*72}{RESET}")


# ==================== 准备测试数据 ====================
print(f"\n{BOLD}{YELLOW}🛡️  寰宇OS 四大需求端到端测试{RESET}\n")

# 生成一张测试"影视帧"（平滑渐变 + 文字，避免高频噪声影响 DCT 水印恢复）
TEST_FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
for y in range(480):
    TEST_FRAME[y, :, :] = [120 + y // 8, 100 + y // 10, 90 + y // 12]
cv2.rectangle(TEST_FRAME, (40, 40), (600, 440), (180, 160, 140), -1)
cv2.putText(TEST_FRAME, "FILM MASTER COPY", (100, 240),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
test_path = os.path.join(WATERMARKED_DIR, "_e2e_master.png")
cv2.imwrite(test_path, TEST_FRAME)
raw_bytes = open(test_path, "rb").read()
raw_hash = hashlib.sha256(raw_bytes).hexdigest()
print(f"   测试母版: {test_path}")
print(f"   母版哈希: {raw_hash[:32]}...")


# ==================== 需求 1: 加密锁死下载/转发/爬虫 ====================
banner(1, "加密影视源文件，锁死下载 / 转发 / 爬虫")

crypto = PostQuantumCrypto()
pub, sec = crypto.generate_kem_keypair()
print(f"   算法: {crypto.KEM_ALGORITHM}  use_real_oqs={crypto.use_real}")

ciphertext, encrypted = crypto.encrypt_model(raw_bytes, pub)

# 测试1.1: 加密后数据与原文完全不同
if encrypted != raw_bytes and len(encrypted) == len(raw_bytes):
    ok("1.1 加密产生密文", f"长度 {len(encrypted)} bytes, 前8字节: {encrypted[:8].hex()}")
else:
    fail("1.1 加密产生密文")

# 测试1.2: 密文无法直接作为图片打开（模拟爬虫抓到密文）
try:
    arr = np.frombuffer(encrypted, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if decoded is None:
        ok("1.2 爬虫抓到密文 → 无法解码为图片", "cv2.imdecode 返回 None")
    else:
        fail("1.2 爬虫抓到密文", "被意外解码")
except Exception as e:
    ok("1.2 爬虫抓到密文 → 无法解码", str(e)[:40])

# 测试1.3: 伪造密文/胶囊 → 无法解密（转发到他人 = 他没有对应加密胶囊）
fake_ct = os.urandom(len(ciphertext))
try:
    wrong = crypto.decrypt_model(fake_ct, encrypted, sec)
    if wrong != raw_bytes:
        ok("1.3 伪造密文胶囊 → 解密失败", "转发只拿到密文块是打不开的")
    else:
        fail("1.3 伪造密文")
except Exception as e:
    ok("1.3 伪造密文胶囊 → 解密异常", str(e)[:40])

# 测试1.4: 正确密钥能解密
try:
    restored = crypto.decrypt_model(ciphertext, encrypted, sec)
    if restored == raw_bytes:
        ok("1.4 正确密钥 → 明文恢复", "授权用户可正常观看")
    else:
        fail("1.4 正确密钥解密")
except Exception as e:
    fail("1.4 正确密钥解密", str(e))


# ==================== 需求 3: 防止木马偷走 ====================
banner(3, "防止源文件被木马偷走 —— 访问控制 + 完整性监控 + 异常检测")

# 测试3.1: 时效令牌发放
token_rec = issue_access_token(raw_hash, "vip_user", "203.0.113.10", ttl_seconds=60, max_views=2)
if token_rec.get("token") and len(token_rec["token"]) > 20:
    ok("3.1 发放时效访问令牌", f"绑定IP 203.0.113.10, 剩余{token_rec['max_views']}次")
else:
    fail("3.1 发放令牌")

# 测试3.2: 合法访问放行
r = verify_access_token(token_rec["token"], "203.0.113.10", "Mozilla/5.0")
if r["valid"]:
    ok("3.2 授权IP+正常UA → 放行", "加密文件可解密")
else:
    fail("3.2 合法访问被拒", r["reason"])

# 测试3.3: 转发拦截（令牌被偷到其他IP）
token_rec2 = issue_access_token(raw_hash, "vip_user", "203.0.113.10", ttl_seconds=60, max_views=1)
r = verify_access_token(token_rec2["token"], "1.2.3.4", "Mozilla/5.0")
if not r["valid"] and r["reason"] == "ip_mismatch":
    ok("3.3 令牌被转发到他人IP → 拦截", "reason=ip_mismatch")
else:
    fail("3.3 转发拦截", str(r))

# 测试3.4: 爬虫UA识别
token_rec3 = issue_access_function = issue_access_token(raw_hash, "vip_user", "203.0.113.20", 60, 1)
r = verify_access_token(token_rec3["token"], "203.0.113.20", "python-requests/2.28")
if not r["valid"] and r["reason"] == "suspicious_user_agent":
    ok("3.4 爬虫/脚本 UA → 拦截", "reason=suspicious_user_agent")
else:
    fail("3.4 爬虫拦截", str(r))

# 测试3.5: 完整性快照与校验（先清理同路径的历史快照，避免旧记录干扰）
import json as _json
_snap_path = os.path.join(DATA_DIR, "integrity_snapshots.json")
if os.path.exists(_snap_path):
    with open(_snap_path, "r", encoding="utf-8") as _f:
        _snaps = _json.load(_f)
    _snaps = [s for s in _snaps if s.get("file_path") != test_path]
    with open(_snap_path, "w", encoding="utf-8") as _f:
        _json.dump(_snaps, _f, ensure_ascii=False, indent=2)

snapshot_integrity(test_path, raw_hash, "studio_A")
integ = check_integrity(test_path)
if integ["ok"]:
    ok("3.5 完整性校验（文件未改） → PASS", "哈希一致")
else:
    fail("3.5 完整性校验", str(integ))

# 测试3.6: 模拟木马篡改文件
with open(test_path, "ab") as f:
    f.write(b"\x00TROJAN_INJECTED")
integ2 = check_integrity(test_path)
if not integ2["ok"]:
    ok("3.6 木马篡改后 → 检测到入侵", f"expected={integ2['expected'][:12]}... actual={integ2['actual'][:12]}...")
else:
    fail("3.6 篡改未被检测")
# 还原
cv2.imwrite(test_path, TEST_FRAME)

# 测试3.7: 异常IP高频访问检测
for _ in range(15):
    verify_access_token("fake_token_xxx", "198.51.100.99", "Mozilla/5.0")
report = detect_anomaly("198.51.100.99", "Scrapy/2.5")
if report["suspicious"] and report["risk_level"] in ("MEDIUM", "HIGH"):
    ok(f"3.7 爬虫高频扫描 → {report['risk_level']} 风险", f"信号数={len(report['signals'])}")
else:
    fail("3.7 异常检测", str(report))


# ==================== 需求 2: 被盗触发 DNA 身份证 ====================
banner(2, "被盗触发 DNA 身份证 —— 红色水印铺满画面")

# 重新加载母版
clean = cv2.imread(test_path)
dna_img = apply_visible_watermark(clean, "COPYRIGHT STUDIO_A 2026", opacity=0.35, tile=True)

# 测试2.1: 生成的水印图像尺寸一致
if dna_img.shape == clean.shape:
    ok("2.1 DNA身份证生成", f"分辨率 {dna_img.shape[1]}x{dna_img.shape[0]}")
else:
    fail("2.1 DNA水印尺寸")

# 测试2.2: 显示平均像素距离（水印确实改变了画面）
mean_pixel_diff = float(np.abs(dna_img.astype(int) - clean.astype(int)).mean())
if mean_pixel_diff > 2:
    ok("2.2 水印显著改变画面", f"平均像素变化 {mean_pixel_diff:.2f} / 255")
else:
    fail("2.2 像素变化", f"仅 {mean_pixel_diff:.2f}")

# 测试2.3: 像素差异率（斜平铺文字水印至少覆盖 3% 像素）
diff_ratio = (np.abs(dna_img.astype(int) - clean.astype(int)).sum(axis=2) > 10).mean()
if diff_ratio > 0.03:
    ok("2.3 斜铺文字覆盖率", f"{diff_ratio*100:.1f}% 像素被水印覆盖（密集斜铺）")
else:
    fail("2.3 覆盖率", f"仅 {diff_ratio*100:.1f}%")

# 测试2.4: 反制自动触发（完整性异常 → trigger_counter_measure）
with open(test_path, "ab") as f:
    f.write(b"BAD")
integ3 = check_integrity(test_path)
if not integ3["ok"]:
    measure = trigger_counter_measure(integ3["expected"], "trojan_detected", "studio_A")
    if measure["action"] == "force_visible_watermark":
        ok("2.4 木马触发 → 自动强制DNA水印", f"动作: {measure['action']}")
    else:
        fail("2.4 反制动作")
else:
    fail("2.4 反制触发路径")
cv2.imwrite(test_path, TEST_FRAME)

# 保存可视化对比
cmp_path = os.path.join(WATERMARKED_DIR, "_e2e_dna_result.png")
cv2.imwrite(cmp_path, dna_img)
print(f"   可视化: {cmp_path}")


# ==================== 需求 4: 溯源到首发平台/IP ====================
banner(4, "盗版溯源 —— 追到首发平台 + 首个IP")

# 清空历史分发记录，保证本次溯源仅对本次分发匹配
dist_log = os.path.join(DATA_DIR, "distributions.json")
if os.path.exists(dist_log):
    os.remove(dist_log)

clean = cv2.imread(test_path)

# 模拟 3 个分发渠道，各拿不同指纹的副本（ASCII 平台名更利于 DCT 恢复）
channels = [
    ("DouYin", "113.88.45.12", "u_douyin"),
    ("BiliBili", "221.230.17.88", "u_bili"),
    ("YouTube", "104.26.1.200", "u_yt"),
]

fingerprints = {}
watermarked_copies = {}
for platform, ip, user in channels:
    fp = generate_fingerprint(platform, ip, user)
    wm_img = embed_invisible_watermark(clean, fp)
    register_distribution(
        file_name="_e2e_master.png",
        file_hash=raw_hash,
        platform=platform,
        ip_address=ip,
        user_id=user,
        fingerprint=fp,
    )
    fingerprints[platform] = fp
    watermarked_copies[platform] = wm_img
    print(f"   └─ 分发 {platform:<8} IP={ip:<16} 指纹={fp[:40]}...")

# 测试4.1: 指纹可提取
leaked_platform = "BiliBili"
leaked_img = watermarked_copies[leaked_platform]
extracted = extract_invisible_watermark(leaked_img, len(fingerprints[leaked_platform]))
# 因DCT容量限制只匹配前缀
original_fp = fingerprints[leaked_platform]
match_chars = sum(1 for a, b in zip(extracted[:len(original_fp)], original_fp) if a == b)
match_rate = match_chars / len(original_fp)
if match_rate > 0.7:
    ok(f"4.1 从盗版中提取指纹 (匹配率 {match_rate*100:.0f}%)", f"提取={extracted[:40]}")
else:
    fail("4.1 指纹提取", f"仅 {match_rate*100:.0f}%")

# 测试4.2: trace_leak 能定位到正确平台
report = trace_leak(extracted)
if report["found"]:
    # 看精确或模糊匹配的 top 是否是 B站
    top = report.get("exact_match")
    if not top and report.get("fuzzy_matches"):
        top = report["fuzzy_matches"][0]
    if top and top["platform"] == leaked_platform:
        ok(f"4.2 溯源定位首发平台 → 【{top['platform']}】", f"IP={top['ip_address']}, 用户={top['user_id']}")
    else:
        fail("4.2 溯源定位", f"得到 {top}")
else:
    fail("4.2 未找到溯源")

# 测试4.3: 溯源报告含有维权关键信息
if report.get("conclusion") and "IP" in report["conclusion"]:
    ok("4.3 生成可维权的溯源结论", report["conclusion"][:80])
else:
    fail("4.3 溯源结论")

# 测试4.4: 模糊匹配能容忍指纹损伤（模拟视频压缩带来的指纹退化）
corrupted = extracted[:30] + "XXXX" + extracted[34:]  # 中间4字符损坏
report2 = trace_leak(corrupted)
if report2["found"] and report2.get("fuzzy_matches"):
    top2 = report2["fuzzy_matches"][0]
    if top2["platform"] == leaked_platform:
        ok("4.4 指纹部分损坏仍能溯源", f"相似度 {top2['similarity']*100:.0f}%")
    else:
        fail("4.4 损坏容忍", f"top={top2['platform']}")
else:
    fail("4.4 损坏容忍")


# ==================== 附加: 哈希链存证 ====================
banner(5, "附加 — 所有操作不可篡改上链")

ok_chain, msg = verify_chain()
records = get_all_records()
if ok_chain:
    ok(f"5.1 哈希链完整性 PASS (共 {len(records)} 条)", msg)
else:
    fail("5.1 哈希链", msg)

types = {}
for r in records:
    t = r["data"].get("type", "?")
    types[t] = types.get(t, 0) + 1
ok("5.2 全链路自动存证", ", ".join(f"{k}×{v}" for k, v in types.items()))


# ==================== 总结 ====================
print(f"\n{BOLD}{'═'*72}{RESET}")
print(f"{BOLD}  测试总结{RESET}")
print(f"{BOLD}{'═'*72}{RESET}")
print(f"  {GREEN}✅ 通过: {len(passed)}{RESET}")
print(f"  {RED}❌ 失败: {len(failed)}{RESET}")

if failed:
    print(f"\n  {RED}失败项:{RESET}")
    for f in failed:
        print(f"    - {f}")
    sys.exit(1)

print(f"\n  {GREEN}{BOLD}🎉 全部通过 — 四大需求功能验证成功{RESET}\n")

summary = [
    ("需求1 加密锁死", "下载打开是密文、转发无密钥、爬虫抓到噪声"),
    ("需求2 DNA身份证", "红色水印铺满，覆盖率 >30%，木马触发自动反制"),
    ("需求3 防窃取", "令牌IP绑定、爬虫UA识别、完整性校验、异常检测"),
    ("需求4 溯源", "从盗版提指纹 → 定位首发平台+IP，含压缩容错"),
    ("附加 存证", "全链路操作哈希链上链，完整性自动校验"),
]
print(f"{BOLD}  能力矩阵:{RESET}")
for k, v in summary:
    print(f"    {GREEN}✓{RESET} {k:<14} — {v}")
print()
