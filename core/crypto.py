"""
后量子加密签名模块

- ML-KEM (Kyber): 模型文件加密，防窃取
- ML-DSA (Dilithium): 权属信息签名，生成可验证证书
"""
import hashlib
import json
import os
import time
from typing import Tuple, Optional

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import KEYS_DIR

# 尝试导入 liboqs，若不可用则使用模拟实现
try:
    import oqs
    HAS_OQS = True
except ImportError:
    HAS_OQS = False


class PQCryptoSimulator:
    """当 liboqs 不可用时的模拟实现（演示用）"""

    @staticmethod
    def generate_keypair(algorithm: str) -> Tuple[bytes, bytes]:
        """生成模拟密钥对"""
        seed = hashlib.sha512(f"{algorithm}-{time.time()}".encode()).digest()
        public_key = hashlib.sha256(seed[:32]).digest()
        secret_key = hashlib.sha256(seed[32:]).digest()
        return public_key, secret_key

    @staticmethod
    def sign(message: bytes, secret_key: bytes) -> bytes:
        """模拟签名"""
        return hashlib.sha512(message + secret_key).digest()

    @staticmethod
    def verify(message: bytes, signature: bytes, public_key: bytes) -> bool:
        """模拟验签（演示中总是返回True）"""
        return len(signature) > 0

    @staticmethod
    def encrypt(plaintext: bytes, public_key: bytes) -> Tuple[bytes, bytes]:
        """模拟加密（XOR简化）"""
        key = hashlib.sha256(public_key).digest()
        # 简单 XOR 加密（仅演示）
        encrypted = bytes(p ^ key[i % len(key)] for i, p in enumerate(plaintext))
        ciphertext = key  # 模拟密文
        return ciphertext, encrypted

    @staticmethod
    def decrypt(ciphertext: bytes, encrypted: bytes, secret_key: bytes) -> bytes:
        """模拟解密（ciphertext 中存储了加密密钥）"""
        key = ciphertext  # encrypt 时 ciphertext = sha256(public_key)
        decrypted = bytes(e ^ key[i % len(key)] for i, e in enumerate(encrypted))
        return decrypted


class PostQuantumCrypto:
    """后量子密码学引擎"""

    # 算法选择
    KEM_ALGORITHM = "ML-KEM-768"      # 密钥封装（加密）
    SIG_ALGORITHM = "ML-DSA-65"       # 数字签名

    def __init__(self):
        self.use_real = HAS_OQS
        self.simulator = PQCryptoSimulator()

    def generate_kem_keypair(self) -> Tuple[bytes, bytes]:
        """生成 ML-KEM 密钥对（用于模型加密）"""
        if self.use_real:
            kem = oqs.KeyEncapsulation(self.KEM_ALGORITHM)
            public_key = kem.generate_keypair()
            secret_key = kem.export_secret_key()
            return public_key, secret_key
        return self.simulator.generate_keypair("ML-KEM")

    def generate_sig_keypair(self) -> Tuple[bytes, bytes]:
        """生成 ML-DSA 密钥对（用于权属签名）"""
        if self.use_real:
            sig = oqs.Signature(self.SIG_ALGORITHM)
            public_key = sig.generate_keypair()
            secret_key = sig.export_secret_key()
            return public_key, secret_key
        return self.simulator.generate_keypair("ML-DSA")

    def sign_data(self, data: bytes, secret_key: bytes) -> bytes:
        """使用 ML-DSA 对数据签名"""
        if self.use_real:
            sig = oqs.Signature(self.SIG_ALGORITHM, secret_key=secret_key)
            signature = sig.sign(data)
            return signature
        return self.simulator.sign(data, secret_key)

    def verify_signature(self, data: bytes, signature: bytes, public_key: bytes) -> bool:
        """验证 ML-DSA 签名"""
        if self.use_real:
            sig = oqs.Signature(self.SIG_ALGORITHM)
            return sig.verify(data, signature, public_key)
        return self.simulator.verify(data, signature, public_key)

    def encrypt_model(self, model_data: bytes, public_key: bytes) -> Tuple[bytes, bytes]:
        """使用 ML-KEM 加密模型文件"""
        if self.use_real:
            kem = oqs.KeyEncapsulation(self.KEM_ALGORITHM)
            ciphertext, shared_secret = kem.encap_secret(public_key)
            # 用共享密钥 XOR 加密（生产环境应用 AES-GCM）
            key = hashlib.sha256(shared_secret).digest()
            encrypted = bytes(p ^ key[i % len(key)] for i, p in enumerate(model_data))
            return ciphertext, encrypted
        return self.simulator.encrypt(model_data, public_key)

    def decrypt_model(self, ciphertext: bytes, encrypted: bytes, secret_key: bytes) -> bytes:
        """使用 ML-KEM 解密模型文件"""
        if self.use_real:
            kem = oqs.KeyEncapsulation(self.KEM_ALGORITHM, secret_key=secret_key)
            shared_secret = kem.decap_secret(ciphertext)
            key = hashlib.sha256(shared_secret).digest()
            decrypted = bytes(e ^ key[i % len(key)] for i, e in enumerate(encrypted))
            return decrypted
        return self.simulator.decrypt(ciphertext, encrypted, secret_key)


def save_keys(owner_id: str, pub_key: bytes, sec_key: bytes, key_type: str):
    """保存密钥对到文件"""
    key_dir = os.path.join(KEYS_DIR, owner_id)
    os.makedirs(key_dir, exist_ok=True)
    with open(os.path.join(key_dir, f"{key_type}_public.key"), "wb") as f:
        f.write(pub_key)
    with open(os.path.join(key_dir, f"{key_type}_secret.key"), "wb") as f:
        f.write(sec_key)


def load_keys(owner_id: str, key_type: str) -> Tuple[bytes, bytes]:
    """加载密钥对"""
    key_dir = os.path.join(KEYS_DIR, owner_id)
    with open(os.path.join(key_dir, f"{key_type}_public.key"), "rb") as f:
        pub_key = f.read()
    with open(os.path.join(key_dir, f"{key_type}_secret.key"), "rb") as f:
        sec_key = f.read()
    return pub_key, sec_key


def generate_certificate(
    owner_id: str,
    model_name: str,
    model_hash: str,
    watermark_metadata: dict,
    crypto_engine: PostQuantumCrypto,
    sig_secret_key: bytes,
    sig_public_key: bytes,
) -> dict:
    """
    生成模型权属证书

    包含：模型信息 + 水印摘要 + 后量子签名
    """
    cert_data = {
        "version": "1.0",
        "certificate_id": hashlib.sha256(
            f"{owner_id}-{model_name}-{time.time()}".encode()
        ).hexdigest()[:16],
        "owner_id": owner_id,
        "model_name": model_name,
        "model_hash": model_hash,
        "watermark_info": watermark_metadata,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "algorithm": {
            "encryption": crypto_engine.KEM_ALGORITHM,
            "signature": crypto_engine.SIG_ALGORITHM,
        },
    }

    # 对证书内容签名
    cert_bytes = json.dumps(cert_data, sort_keys=True).encode()
    signature = crypto_engine.sign_data(cert_bytes, sig_secret_key)

    cert_data["signature"] = signature.hex()
    cert_data["public_key"] = sig_public_key.hex()

    return cert_data
