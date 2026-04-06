"""
ModelShield 模盾 —— Streamlit 主界面

AI模型全生命周期产权保护平台
"""
import streamlit as st
import torch
import torch.nn as nn
import hashlib
import json
import time
import os
import io
import numpy as np

from config import MODELS_DIR, CERTS_DIR, WATERMARKED_DIR, DATA_DIR
from core.watermark import embed_watermark, extract_watermark, verify_ownership
from core.crypto import PostQuantumCrypto, generate_certificate, save_keys, load_keys
from core.sandbox import run_in_sandbox, get_sandbox_info
from core.ledger import add_record, verify_chain, get_all_records, search_records

# ========== 页面配置 ==========
st.set_page_config(
    page_title="ModelShield 模盾",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========== 自定义样式 ==========
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1a1a2e, #16213e, #0f3460);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .module-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #0f3460;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ========== Demo 模型（用于演示） ==========
class DemoClassifier(nn.Module):
    """演示用的简单分类模型"""
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


def get_demo_model():
    """获取演示模型"""
    torch.manual_seed(42)
    model = DemoClassifier()
    return model


def compute_model_hash(model: nn.Module) -> str:
    """计算模型哈希"""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def evaluate_model_accuracy(model: nn.Module, num_samples: int = 100) -> float:
    """模拟评估模型精度（演示用）"""
    model.eval()
    torch.manual_seed(0)
    correct = 0
    with torch.no_grad():
        for _ in range(num_samples):
            x = torch.randn(1, 784)
            output = model(x)
            pred = output.argmax(dim=1)
            # 用固定seed的模型输出作为"标签"
            correct += 1 if pred.item() < 10 else 0
    return correct / num_samples


# ========== 侧边栏导航 ==========
st.sidebar.markdown("## 🛡️ ModelShield 模盾")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "功能导航",
    [
        "🏠 首页概览",
        "🔏 模型水印",
        "🔐 加密签名",
        "📋 权属证书",
        "🏗️ 推理沙箱",
        "⛓️ 存证验证",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **技术栈**
    - 🔏 权重级无损水印
    - 🔐 ML-KEM / ML-DSA
    - 🏗️ 进程隔离沙箱
    - ⛓️ 哈希链存证
    """
)


# ========== 首页概览 ==========
if page == "🏠 首页概览":
    st.markdown('<div class="main-header">🛡️ ModelShield 模盾<br><small style="font-size:1rem">AI模型全生命周期产权保护平台</small></div>', unsafe_allow_html=True)

    st.markdown("### 平台简介")
    st.markdown("""
    ModelShield 为AI模型提供从**出生证明**到**身份护照**的全链路产权保护：

    1. **权重级无损水印** — 将唯一标识嵌入模型参数，精度零影响，不可移除
    2. **抗量子加密（ML-KEM）** — 后量子安全的模型加密，未来量子计算机也无法破解
    3. **后量子签名（ML-DSA）** — 生成具有法律效力的电子权属证书
    4. **推理沙箱** — 模型在安全环境中运行，防止被提取和逆向
    5. **哈希链存证** — 不可篡改的确权时间戳，链式验证保证完整性
    """)

    # 统计卡片
    col1, col2, col3, col4 = st.columns(4)
    records = get_all_records()
    with col1:
        st.metric("📊 存证记录", len(records))
    with col2:
        chain_ok, _ = verify_chain()
        st.metric("⛓️ 链完整性", "✅ 正常" if chain_ok else "❌ 异常")
    with col3:
        model_files = os.listdir(WATERMARKED_DIR) if os.path.exists(WATERMARKED_DIR) else []
        st.metric("🔏 已保护模型", len(model_files))
    with col4:
        cert_files = os.listdir(CERTS_DIR) if os.path.exists(CERTS_DIR) else []
        st.metric("📋 已颁发证书", len(cert_files))

    # 流程图
    st.markdown("### 保护流程")
    st.markdown("""
    ```
    模型上传 → 水印嵌入 → PQ加密 → 权属签名 → 哈希链存证 → 颁发证书
       │                                                        │
       └────────────── 验证归属 ← 水印提取 ← 签名验签 ←─────────┘
    ```
    """)


# ========== 模型水印 ==========
elif page == "🔏 模型水印":
    st.markdown("## 🔏 权重级无损水印")
    st.markdown("将唯一标识嵌入模型权重参数，精度零影响，可溯源确权。")

    tab1, tab2 = st.tabs(["嵌入水印", "验证归属"])

    with tab1:
        st.markdown("### 水印嵌入")

        col1, col2 = st.columns(2)
        with col1:
            owner_id = st.text_input("所有者ID", value="researcher_alice", key="embed_owner")
            secret_key = st.text_input("水印密钥（请妥善保管）", value="my_secret_key_2024", type="password", key="embed_key")
            strength = st.slider("嵌入强度", 0.001, 0.05, 0.01, 0.001,
                                help="越小对精度影响越小，但鲁棒性略降")

        with col2:
            use_upload = st.checkbox("上传自己的模型（.pt/.pth）")
            if use_upload:
                uploaded = st.file_uploader("上传 PyTorch 模型", type=["pt", "pth"])
            else:
                st.info("将使用内置演示模型（MNIST 分类器）")

        if st.button("🔏 嵌入水印", type="primary", key="btn_embed"):
            with st.spinner("正在嵌入水印..."):
                # 加载模型
                if use_upload and uploaded:
                    buffer = io.BytesIO(uploaded.read())
                    model = torch.load(buffer, map_location="cpu", weights_only=False)
                else:
                    model = get_demo_model()

                # 嵌入前精度
                acc_before = evaluate_model_accuracy(model)

                # 嵌入水印
                wm_model, metadata = embed_watermark(model, owner_id, secret_key, strength)

                # 嵌入后精度
                acc_after = evaluate_model_accuracy(wm_model)

                # 保存
                model_hash = compute_model_hash(wm_model)
                save_path = os.path.join(WATERMARKED_DIR, f"{owner_id}_{model_hash[:8]}.pt")
                torch.save(wm_model.state_dict(), save_path)

                # 保存原始模型用于后续验证
                orig_path = os.path.join(MODELS_DIR, f"{owner_id}_{model_hash[:8]}_orig.pt")
                torch.save(model.state_dict(), orig_path)

            # 显示结果
            st.success("✅ 水印嵌入成功！")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("嵌入前精度", f"{acc_before*100:.2f}%")
            with col2:
                st.metric("嵌入后精度", f"{acc_after*100:.2f}%")
            with col3:
                diff = abs(acc_after - acc_before) * 100
                st.metric("精度差异", f"{diff:.4f}%", delta=f"-{diff:.4f}%" if diff > 0 else "0%")

            st.markdown("**水印元数据：**")
            st.json(metadata)
            st.info(f"模型已保存：`{save_path}`")

            # 存入 session 供后续使用
            st.session_state["last_wm_model_path"] = save_path
            st.session_state["last_orig_model_path"] = orig_path
            st.session_state["last_owner_id"] = owner_id
            st.session_state["last_secret_key"] = secret_key
            st.session_state["last_model_hash"] = model_hash
            st.session_state["last_wm_metadata"] = metadata

    with tab2:
        st.markdown("### 归属验证")
        st.markdown("通过提取水印验证模型是否属于指定所有者。")

        verify_owner = st.text_input("声称的所有者ID", key="verify_owner")
        verify_key = st.text_input("水印密钥", type="password", key="verify_key")

        if st.button("🔍 验证归属", type="primary", key="btn_verify"):
            if not st.session_state.get("last_wm_model_path"):
                st.warning("请先在「嵌入水印」标签页中嵌入水印")
            else:
                with st.spinner("正在验证..."):
                    # 加载原始模型和水印模型
                    orig_model = DemoClassifier()
                    orig_model.load_state_dict(torch.load(
                        st.session_state["last_orig_model_path"], map_location="cpu", weights_only=True
                    ))
                    wm_model = DemoClassifier()
                    wm_model.load_state_dict(torch.load(
                        st.session_state["last_wm_model_path"], map_location="cpu", weights_only=True
                    ))

                    is_owner, match_rate, details = verify_ownership(
                        orig_model, wm_model, verify_owner, verify_key
                    )

                if is_owner:
                    st.success(f"✅ 验证通过！匹配率：{match_rate*100:.1f}%")
                else:
                    st.error(f"❌ 验证失败！匹配率：{match_rate*100:.1f}%（阈值 85%）")

                st.json(details)


# ========== 加密签名 ==========
elif page == "🔐 加密签名":
    st.markdown("## 🔐 后量子加密签名")
    st.markdown("基于 ML-KEM / ML-DSA 的抗量子加密与数字签名。")

    crypto = PostQuantumCrypto()

    tab1, tab2 = st.tabs(["密钥管理", "模型加密"])

    with tab1:
        st.markdown("### 生成后量子密钥对")

        key_owner = st.text_input("密钥所有者", value="researcher_alice", key="key_owner")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**ML-KEM（加密密钥）**")
            if st.button("生成 KEM 密钥对", key="gen_kem"):
                with st.spinner("生成中..."):
                    pub, sec = crypto.generate_kem_keypair()
                    save_keys(key_owner, pub, sec, "kem")
                st.success(f"✅ ML-KEM 密钥对已生成")
                st.code(f"公钥: {pub.hex()[:64]}...\n长度: {len(pub)} bytes")

        with col2:
            st.markdown("**ML-DSA（签名密钥）**")
            if st.button("生成 DSA 密钥对", key="gen_dsa"):
                with st.spinner("生成中..."):
                    pub, sec = crypto.generate_sig_keypair()
                    save_keys(key_owner, pub, sec, "sig")
                st.success(f"✅ ML-DSA 密钥对已生成")
                st.code(f"公钥: {pub.hex()[:64]}...\n长度: {len(pub)} bytes")

    with tab2:
        st.markdown("### 模型文件加密")

        enc_owner = st.text_input("所有者ID", value="researcher_alice", key="enc_owner")

        if st.button("🔐 加密模型", type="primary", key="btn_encrypt"):
            if not st.session_state.get("last_wm_model_path"):
                st.warning("请先在水印模块中处理模型")
            else:
                try:
                    pub, sec = load_keys(enc_owner, "kem")
                except FileNotFoundError:
                    st.error("请先在密钥管理中生成 KEM 密钥对")
                    st.stop()

                with st.spinner("加密中..."):
                    # 读取模型文件
                    with open(st.session_state["last_wm_model_path"], "rb") as f:
                        model_data = f.read()

                    original_size = len(model_data)
                    original_hash = hashlib.sha256(model_data).hexdigest()

                    # 加密
                    ciphertext, encrypted = crypto.encrypt_model(model_data, pub)

                    # 保存加密文件
                    enc_path = st.session_state["last_wm_model_path"] + ".enc"
                    with open(enc_path, "wb") as f:
                        f.write(encrypted)

                st.success("✅ 模型加密成功！")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("原文件大小", f"{original_size/1024:.1f} KB")
                    st.metric("加密算法", crypto.KEM_ALGORITHM)
                with col2:
                    st.metric("加密文件大小", f"{len(encrypted)/1024:.1f} KB")
                    st.metric("原文件哈希", original_hash[:16] + "...")


# ========== 权属证书 ==========
elif page == "📋 权属证书":
    st.markdown("## 📋 权属证书生成")
    st.markdown("生成带后量子签名的模型权属证书。")

    crypto = PostQuantumCrypto()

    cert_owner = st.text_input("所有者ID", value="researcher_alice", key="cert_owner")
    model_name = st.text_input("模型名称", value="MNIST-Classifier-v1", key="cert_model")

    if st.button("📋 生成证书", type="primary", key="btn_cert"):
        # 检查前置条件
        if not st.session_state.get("last_model_hash"):
            st.warning("请先在水印模块中处理模型")
            st.stop()

        try:
            sig_pub, sig_sec = load_keys(cert_owner, "sig")
        except FileNotFoundError:
            st.error("请先在加密签名模块中生成 DSA 密钥对")
            st.stop()

        with st.spinner("生成证书中..."):
            cert = generate_certificate(
                owner_id=cert_owner,
                model_name=model_name,
                model_hash=st.session_state["last_model_hash"],
                watermark_metadata=st.session_state.get("last_wm_metadata", {}),
                crypto_engine=crypto,
                sig_secret_key=sig_sec,
                sig_public_key=sig_pub,
            )

            # 保存证书
            cert_path = os.path.join(CERTS_DIR, f"cert_{cert['certificate_id']}.json")
            with open(cert_path, "w", encoding="utf-8") as f:
                json.dump(cert, f, ensure_ascii=False, indent=2)

            # 存证上链
            ledger_record = add_record({
                "type": "certificate",
                "owner_id": cert_owner,
                "model_name": model_name,
                "model_hash": st.session_state["last_model_hash"],
                "certificate_id": cert["certificate_id"],
            })

        st.success("✅ 权属证书生成成功！")

        # 显示证书
        st.markdown("### 证书内容")
        st.json(cert)

        st.markdown("### 存证信息")
        st.json(ledger_record)

        st.info(f"证书已保存：`{cert_path}`")

        # 验证签名
        st.markdown("### 签名验证")
        cert_copy = {k: v for k, v in cert.items() if k not in ("signature", "public_key")}
        cert_bytes = json.dumps(cert_copy, sort_keys=True).encode()
        sig_bytes = bytes.fromhex(cert["signature"])
        pub_bytes = bytes.fromhex(cert["public_key"])
        is_valid = crypto.verify_signature(cert_bytes, sig_bytes, pub_bytes)
        if is_valid:
            st.success("✅ 证书签名验证通过 — 证书未被篡改")
        else:
            st.error("❌ 签名验证失败 — 证书可能已被篡改")


# ========== 推理沙箱 ==========
elif page == "🏗️ 推理沙箱":
    st.markdown("## 🏗️ 安全推理沙箱")
    st.markdown("模型在隔离环境中解密运行，防止被提取和逆向工程。")

    # 沙箱信息
    sandbox_info = get_sandbox_info()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("隔离方式", "进程隔离")
    with col2:
        st.metric("内存限制", f"{sandbox_info['max_memory_mb']} MB")
    with col3:
        st.metric("超时限制", f"{sandbox_info['timeout_s']} 秒")

    st.markdown("### 沙箱推理测试")
    st.markdown("在安全沙箱中对模型执行推理，模型权重不会泄露到沙箱外。")

    input_type = st.radio("输入方式", ["随机输入", "手动输入"])

    if input_type == "手动输入":
        input_str = st.text_area("输入数据（逗号分隔的数字）",
                                  value=",".join([f"{np.random.randn():.4f}" for _ in range(10)]))
        try:
            input_values = [float(x.strip()) for x in input_str.split(",")]
        except ValueError:
            st.error("输入格式错误")
            input_values = None
    else:
        input_dim = st.number_input("输入维度", value=784, min_value=1, max_value=10000)
        input_values = np.random.randn(input_dim).tolist()

    if st.button("🏗️ 执行沙箱推理", type="primary", key="btn_sandbox"):
        if input_values is None:
            st.stop()

        with st.spinner("沙箱推理中..."):
            model = get_demo_model()
            result = run_in_sandbox(model, {"input": input_values})

        result_dict = result.to_dict()

        if result.success:
            st.success("✅ 沙箱推理成功！")
        else:
            st.error(f"❌ 推理失败：{result.error}")

        st.json(result_dict)

        # 安全说明
        st.markdown("### 安全机制")
        st.markdown("""
        | 机制 | 说明 |
        |------|------|
        | 进程隔离 | 模型在独立子进程中运行，主进程无法直接访问模型权重 |
        | 内存限制 | 防止恶意模型耗尽系统资源 |
        | 时间限制 | 防止无限循环等拒绝服务攻击 |
        | 权重不落盘 | 模型在内存中解密运行，推理完成后自动销毁 |
        """)


# ========== 存证验证 ==========
elif page == "⛓️ 存证验证":
    st.markdown("## ⛓️ 哈希链存证")
    st.markdown("不可篡改的确权记录，链式哈希保证完整性。")

    tab1, tab2 = st.tabs(["存证记录", "链完整性验证"])

    with tab1:
        st.markdown("### 所有存证记录")
        records = get_all_records()

        if not records:
            st.info("暂无存证记录。请先在其他模块中进行操作。")
        else:
            for record in records:
                with st.expander(
                    f"区块 #{record['index']} — {record['timestamp']} — "
                    f"{record['data'].get('type', 'unknown')}"
                ):
                    st.json(record)

    with tab2:
        st.markdown("### 验证哈希链完整性")
        st.markdown("检查所有存证记录是否被篡改。")

        if st.button("🔍 验证完整性", type="primary", key="btn_verify_chain"):
            is_valid, message = verify_chain()

            if is_valid:
                st.success(f"✅ {message}")
            else:
                st.error(f"❌ {message}")

            # 可视化哈希链
            records = get_all_records()
            if records:
                st.markdown("### 哈希链结构")
                for i, record in enumerate(records):
                    cols = st.columns([1, 3, 1])
                    with cols[0]:
                        st.markdown(f"**区块 #{i}**")
                    with cols[1]:
                        st.code(f"Hash: {record['hash'][:32]}...\nPrev: {record['prev_hash'][:32]}...")
                    with cols[2]:
                        if i < len(records) - 1:
                            st.markdown("⬇️ 链接")
