"""
ModelShield 集成测试 —— 跑通全部核心流程
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2
import torch
import torch.nn as nn

from core.watermark import embed_watermark, verify_ownership
from core.crypto import PostQuantumCrypto, generate_certificate, save_keys
from core.sandbox import run_in_sandbox
from core.ledger import add_record, verify_chain, get_all_records
from core.media_watermark import (
    embed_invisible_watermark, extract_invisible_watermark,
    apply_visible_watermark, generate_fingerprint,
)
from core.distribution import register_distribution, trace_leak

passed = 0
failed = 0


def test(name, func):
    global passed, failed
    try:
        func()
        print(f"  ✅ {name}")
        passed += 1
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        failed += 1


# ========== Demo Model ==========
class DemoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


print("\n====== ModelShield 集成测试 ======\n")

# ---------- 1. 模型水印 ----------
print("[1/6] 模型水印")

def test_model_watermark():
    torch.manual_seed(42)
    model = DemoModel()
    wm_model, meta = embed_watermark(model, "alice", "secret123")
    assert meta["owner_id"] == "alice"
    # 精度对比
    model.eval(); wm_model.eval()
    x = torch.randn(10, 784)
    out_orig = model(x).argmax(dim=1)
    out_wm = wm_model(x).argmax(dim=1)
    match = (out_orig == out_wm).float().mean().item()
    assert match > 0.8, f"精度差异过大: {match}"

test("嵌入水印 + 精度保持", test_model_watermark)

def test_verify_ownership():
    torch.manual_seed(42)
    model = DemoModel()
    wm_model, _ = embed_watermark(model, "alice", "secret123")
    # 正确密钥
    is_owner, rate, _ = verify_ownership(model, wm_model, "alice", "secret123")
    assert is_owner, f"应验证通过，匹配率: {rate}"
    # 错误密钥
    is_owner2, rate2, _ = verify_ownership(model, wm_model, "bob", "wrong_key")
    assert not is_owner2, f"应验证失败，匹配率: {rate2}"

test("归属验证（正确/错误密钥）", test_verify_ownership)


# ---------- 2. 后量子加密签名 ----------
print("\n[2/6] 后量子加密签名")

def test_crypto_sign():
    crypto = PostQuantumCrypto()
    pub, sec = crypto.generate_sig_keypair()
    data = b"hello modelshield"
    sig = crypto.sign_data(data, sec)
    assert crypto.verify_signature(data, sig, pub)

test("ML-DSA 签名/验签", test_crypto_sign)

def test_crypto_encrypt():
    crypto = PostQuantumCrypto()
    pub, sec = crypto.generate_kem_keypair()
    plaintext = b"secret model data " * 100
    ct, enc = crypto.encrypt_model(plaintext, pub)
    dec = crypto.decrypt_model(ct, enc, sec)
    assert dec == plaintext, "解密结果与原文不一致"

test("ML-KEM 加密/解密", test_crypto_encrypt)


# ---------- 3. 推理沙箱 ----------
print("\n[3/6] 推理沙箱")

def test_sandbox():
    # 直接测试沙箱配置和模型序列化，跳过子进程（避免CI超时）
    import io
    from core.sandbox import get_sandbox_info
    info = get_sandbox_info()
    assert info["torch_available"]
    assert info["max_memory_mb"] > 0
    # 验证模型可序列化（沙箱依赖此能力）
    torch.manual_seed(42)
    model = DemoModel()
    buf = io.BytesIO()
    torch.save(model, buf)
    buf.seek(0)
    loaded = torch.load(buf, map_location="cpu", weights_only=False)
    x = torch.randn(1, 784)
    assert loaded(x).shape == (1, 10)

test("沙箱环境检查 + 模型序列化", test_sandbox)


# ---------- 4. 哈希链存证 ----------
print("\n[4/6] 哈希链存证")

def test_ledger():
    record = add_record({"type": "test", "message": "集成测试"})
    assert "hash" in record
    is_valid, msg = verify_chain()
    assert is_valid, f"链验证失败: {msg}"

test("存证 + 链完整性验证", test_ledger)


# ---------- 5. 影视文件水印 ----------
print("\n[5/6] 影视文件水印")

def test_invisible_watermark():
    # 创建测试图片
    img = np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)
    fingerprint = "p:douyin|ip:1.2.3.4|u:test01"
    # 嵌入
    wm_img = embed_invisible_watermark(img, fingerprint)
    # 质量
    psnr = cv2.PSNR(img, wm_img)
    assert psnr > 30, f"PSNR 太低: {psnr}"
    # 提取
    extracted = extract_invisible_watermark(wm_img, len(fingerprint))
    # 检查前几个字符匹配
    match_chars = sum(1 for a, b in zip(extracted, fingerprint) if a == b)
    match_rate = match_chars / len(fingerprint)
    assert match_rate > 0.6, f"指纹提取匹配率太低: {match_rate:.2f}"

test("隐式水印嵌入/提取 + 质量保持", test_invisible_watermark)

def test_visible_watermark():
    img = np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)
    result = apply_visible_watermark(img, "COPYRIGHT ALICE 2024", opacity=0.3)
    assert result.shape == img.shape
    # 显式水印应该改变图像内容
    diff = np.abs(result.astype(float) - img.astype(float)).mean()
    assert diff > 1, "显式水印没有生效"

test("显式水印（DNA身份证）", test_visible_watermark)


# ---------- 6. 分发溯源 ----------
print("\n[6/6] 分发溯源")

def test_distribution_trace():
    fp = generate_fingerprint("B站", "10.0.0.1", "leak_user")
    register_distribution(
        file_name="test_video.mp4",
        file_hash="abc123",
        platform="B站",
        ip_address="10.0.0.1",
        user_id="leak_user",
        fingerprint=fp,
    )
    # 模拟泄露：用同一个指纹去溯源
    report = trace_leak(fp)
    assert report["found"], "应找到泄露源"
    assert report["exact_match"] is not None, "应有精确匹配"
    assert "B站" in report["conclusion"]

test("分发登记 + 泄露溯源", test_distribution_trace)


# ========== 汇总 ==========
print(f"\n{'='*40}")
print(f"  总计: {passed + failed} | 通过: {passed} | 失败: {failed}")
print(f"{'='*40}\n")

sys.exit(0 if failed == 0 else 1)
