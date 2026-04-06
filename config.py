"""ModelShield 全局配置"""
import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据存储目录
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(DATA_DIR, "models")          # 上传的模型文件
CERTS_DIR = os.path.join(DATA_DIR, "certs")             # 生成的证书
WATERMARKED_DIR = os.path.join(DATA_DIR, "watermarked") # 加水印后的模型
LEDGER_PATH = os.path.join(DATA_DIR, "ledger.json")     # 哈希链存证账本
KEYS_DIR = os.path.join(DATA_DIR, "keys")               # PQ密钥对

# 确保目录存在
for d in [DATA_DIR, MODELS_DIR, CERTS_DIR, WATERMARKED_DIR, KEYS_DIR]:
    os.makedirs(d, exist_ok=True)

# 水印配置
WATERMARK_STRENGTH = 0.01  # 水印嵌入强度（越小对精度影响越小）
WATERMARK_KEY_LENGTH = 64  # 水印密钥长度（bits）

# 沙箱配置
SANDBOX_TIMEOUT = 30  # 推理超时（秒）
SANDBOX_MAX_MEMORY = 512 * 1024 * 1024  # 512MB 内存限制
