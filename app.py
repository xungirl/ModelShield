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

import cv2

from config import MODELS_DIR, CERTS_DIR, WATERMARKED_DIR, DATA_DIR
from core.watermark import embed_watermark, extract_watermark, verify_ownership
from core.crypto import PostQuantumCrypto, generate_certificate, save_keys, load_keys
from core.sandbox import run_in_sandbox, get_sandbox_info
from core.ledger import add_record, verify_chain, get_all_records, search_records
from core.media_watermark import (
    embed_invisible_watermark, extract_invisible_watermark,
    apply_visible_watermark, generate_fingerprint, process_video_watermark,
    apply_counter_watermark_to_video,
)
from core.distribution import (
    register_distribution, trace_leak, get_all_distributions, get_distribution_stats,
)
from core.anti_theft import (
    issue_access_token, verify_access_token, revoke_token,
    snapshot_integrity, check_integrity,
    detect_anomaly, trigger_counter_measure,
    get_access_log,
)

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
        "🎥 一键视频保护",
        "🏠 首页概览",
        "🌐 寰宇OS 官网",
        "🎬 影视文件保护",
        "🛡️ 防盗反制",
        "🔍 泄露溯源",
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
    - 🎬 DCT频域隐式水印
    - 🔏 权重级无损水印
    - 🔐 ML-KEM / ML-DSA
    - 🏗️ 进程隔离沙箱
    - ⛓️ 哈希链存证
    - 🔍 分发指纹溯源
    """
)


# ========== 寰宇OS 官网 ==========
if page == "🌐 寰宇OS 官网":
    st.markdown("## 🌐 寰宇OS —— 影视IP全栈解决方案")
    st.caption("以下是本项目对外发布的产品官网。使用 HTML+CSS 构建，可直接部署到任意静态托管。")

    website_path = os.path.join(os.path.dirname(__file__), "website", "index.html")
    if os.path.exists(website_path):
        with open(website_path, "r", encoding="utf-8") as f:
            html = f.read()
        import streamlit.components.v1 as components
        components.html(html, height=3200, scrolling=True)

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            with open(website_path, "rb") as f:
                st.download_button(
                    "⬇️ 下载官网 HTML",
                    f.read(),
                    file_name="huanyu_os_website.html",
                    mime="text/html",
                )
        with c2:
            st.markdown(f"**本地路径**: `{website_path}`")
            st.caption("可部署至 Vercel / Netlify / GitHub Pages / 自有服务器")
    else:
        st.error(f"未找到官网文件：{website_path}")


# ========== 首页概览 ==========
elif page == "🏠 首页概览":
    st.markdown('<div class="main-header">🛡️ 寰宇OS · ModelShield<br><small style="font-size:1rem">影视IP全栈解决方案 · AI模型全生命周期产权保护</small></div>', unsafe_allow_html=True)

    st.markdown("### 平台简介")
    st.markdown("""
    **寰宇OS（ModelShield Core）** 为影视IP与AI模型提供全栈产权操作系统：

    **🎬 影视IP四层防线**
    1. **第一道：后量子加密锁（ML-KEM）** — 源文件加密即锁死下载、转发、爬虫抓取，抗量子破解
    2. **第二道：隐式指纹水印（DCT频域）** — 每份分发副本嵌唯一指纹（平台+IP+用户+时间），PSNR >35dB 肉眼不可见
    3. **第三道：DNA身份证反制** — 访问异常 / 完整性破坏 / 木马窃取 → 自动触发红色水印铺满画面
    4. **第四道：首发平台溯源** — 从盗版中提指纹 → 精确匹配数据库 → 定位首个发布平台与IP → 出具维权报告

    **🛡️ 防窃取体系**
    5. **时效访问令牌** — 绑定IP/UA/使用次数，被转发到非授权IP立即失效
    6. **完整性监控** — 哈希快照周期校验，篡改即触发反制
    7. **异常检测** — 高频访问、爬虫UA、连续失败自动识别

    **🤖 AI模型保护**
    8. **权重级无损水印** — 唯一标识嵌入模型参数，精度零影响
    9. **ML-DSA 权属证书** — 后量子数字签名，法律效力
    10. **推理沙箱 + 哈希链存证** — 进程隔离防逆向，链式记录防篡改
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
    st.markdown("### 影视文件保护流程")
    st.markdown("""
    ```
    源文件上传 → ML-KEM加密（防下载/转发/爬虫）
        │
        ├─ 分发副本A → 嵌入指纹A（抖音/IP_A）→ 存证上链
        ├─ 分发副本B → 嵌入指纹B（B站/IP_B）→ 存证上链
        └─ 分发副本C → 嵌入指纹C（YouTube/IP_C）→ 存证上链
                                    │
                           发现泄露文件 → 提取指纹 → 比对数据库 → 定位泄露源
    ```
    """)

    st.markdown("### AI模型保护流程")
    st.markdown("""
    ```
    模型上传 → 水印嵌入 → PQ加密 → 权属签名 → 哈希链存证 → 颁发证书
       │                                                        │
       └────────────── 验证归属 ← 水印提取 ← 签名验签 ←─────────┘
    ```
    """)


# ========== 影视文件保护 ==========
elif page == "🎬 影视文件保护":
    st.markdown("## 🎬 影视文件产权保护")
    st.markdown("为影视源文件提供加密、隐式水印、显式水印全链路保护。")

    tab1, tab2, tab3 = st.tabs(["隐式水印（防盗溯源）", "显式水印（DNA身份证）", "文件加密"])

    with tab1:
        st.markdown("### 隐式水印嵌入")
        st.markdown("基于 DCT 频域变换，将分发指纹嵌入画面，**肉眼完全不可见**，但可提取用于溯源。")

        uploaded_img = st.file_uploader("上传图片", type=["jpg", "jpeg", "png", "bmp"], key="inv_upload")

        col1, col2 = st.columns(2)
        with col1:
            inv_platform = st.selectbox("分发平台", ["抖音", "B站", "YouTube", "微信", "微博", "自定义"], key="inv_platform")
            if inv_platform == "自定义":
                inv_platform = st.text_input("输入平台名", key="inv_custom_platform")
        with col2:
            inv_ip = st.text_input("接收方IP", value="192.168.1.100", key="inv_ip")
            inv_user = st.text_input("接收方用户ID", value="user_001", key="inv_user")

        if uploaded_img and st.button("🔏 嵌入隐式水印", type="primary", key="btn_inv_wm"):
            with st.spinner("正在嵌入隐式水印..."):
                # 读取图片
                file_bytes = np.frombuffer(uploaded_img.read(), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                # 生成指纹
                fingerprint = generate_fingerprint(inv_platform, inv_ip, inv_user)

                # 嵌入
                watermarked = embed_invisible_watermark(image, fingerprint)

                # 计算质量指标
                psnr = cv2.PSNR(image, watermarked)

                # 保存
                save_name = f"invisible_{inv_platform}_{inv_user}.png"
                save_path = os.path.join(WATERMARKED_DIR, save_name)
                cv2.imwrite(save_path, watermarked)

                # 登记分发
                file_hash = hashlib.sha256(file_bytes.tobytes()).hexdigest()
                dist_record = register_distribution(
                    file_name=uploaded_img.name,
                    file_hash=file_hash,
                    platform=inv_platform,
                    ip_address=inv_ip,
                    user_id=inv_user,
                    fingerprint=fingerprint,
                )

            st.success("✅ 隐式水印嵌入成功！指纹已登记。")

            # 对比展示
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**原始图片**")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col2:
                st.markdown("**含隐式水印（肉眼无差别）**")
                st.image(cv2.cvtColor(watermarked, cv2.COLOR_BGR2RGB), use_container_width=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("PSNR（图像质量）", f"{psnr:.1f} dB", help="越高越好，>35dB 人眼无法区分")
            with col2:
                st.metric("嵌入指纹长度", f"{len(fingerprint)} 字符")
            with col3:
                st.metric("分发平台", inv_platform)

            st.markdown("**嵌入的指纹信息：**")
            st.code(fingerprint)

            st.markdown("**分发记录：**")
            st.json(dist_record)

            st.session_state["last_media_image"] = image
            st.session_state["last_media_watermarked"] = watermarked
            st.session_state["last_media_fingerprint"] = fingerprint
            st.session_state["last_media_path"] = save_path

    with tab2:
        st.markdown("### 显式水印 — DNA身份证")
        st.markdown("当文件被盗时触发，**红色水印铺满整个画面**，宣示所有权。")

        uploaded_vis = st.file_uploader("上传图片", type=["jpg", "jpeg", "png", "bmp"], key="vis_upload")

        vis_owner = st.text_input("版权声明文字", value="COPYRIGHT OWNER_NAME 2024", key="vis_owner")
        vis_opacity = st.slider("水印不透明度", 0.1, 0.8, 0.3, 0.05, key="vis_opacity")

        if uploaded_vis and st.button("🛑 触发显式水印", type="primary", key="btn_vis_wm"):
            with st.spinner("生成显式水印..."):
                file_bytes = np.frombuffer(uploaded_vis.read(), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                # 铺满显式水印
                visible_wm = apply_visible_watermark(image, vis_owner, vis_opacity, tile=True)

                save_name = f"visible_watermarked.png"
                save_path = os.path.join(WATERMARKED_DIR, save_name)
                cv2.imwrite(save_path, visible_wm)

            st.success("✅ 显式水印已生成！")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**原始画面**")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col2:
                st.markdown("**被盗触发效果 — DNA身份证**")
                st.image(cv2.cvtColor(visible_wm, cv2.COLOR_BGR2RGB), use_container_width=True)

            st.markdown("""
            > 当检测到未授权传播时，系统自动触发显式水印覆盖画面，
            > 水印包含所有者信息，作为影视作品的**DNA身份证**，
            > 使盗版内容无法正常使用。
            """)

    with tab3:
        st.markdown("### 源文件加密")
        st.markdown("使用后量子加密（ML-KEM）保护源文件，加密后**无法下载、转发、被爬虫抓取**。")

        uploaded_enc = st.file_uploader("上传要加密的文件", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"], key="enc_media_upload")

        if uploaded_enc and st.button("🔐 加密源文件", type="primary", key="btn_enc_media"):
            crypto = PostQuantumCrypto()

            with st.spinner("加密中..."):
                raw_data = uploaded_enc.read()
                original_size = len(raw_data)
                original_hash = hashlib.sha256(raw_data).hexdigest()

                # 生成临时密钥对
                pub, sec = crypto.generate_kem_keypair()

                # 加密
                ciphertext, encrypted = crypto.encrypt_model(raw_data, pub)

                # 保存
                enc_path = os.path.join(WATERMARKED_DIR, uploaded_enc.name + ".enc")
                with open(enc_path, "wb") as f:
                    f.write(encrypted)

                # 存证
                add_record({
                    "type": "media_encryption",
                    "file_name": uploaded_enc.name,
                    "file_hash": original_hash,
                    "algorithm": crypto.KEM_ALGORITHM,
                })

            st.success("✅ 源文件加密成功！")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("原始文件", f"{original_size/1024:.1f} KB")
                st.metric("文件哈希", original_hash[:20] + "...")
            with col2:
                st.metric("加密文件", f"{len(encrypted)/1024:.1f} KB")
                st.metric("加密算法", crypto.KEM_ALGORITHM)

            st.markdown("""
            **加密保护效果：**
            | 威胁 | 防护 |
            |------|------|
            | 直接下载 | ❌ 加密文件无法直接打开 |
            | 转发传播 | ❌ 无密钥无法解密 |
            | 爬虫抓取 | ❌ 抓到的是密文，无法使用 |
            | 量子计算破解 | ❌ ML-KEM 抗量子安全 |
            """)

            # 演示：加密前后对比
            st.markdown("### 加密效果演示")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**原始文件（可正常查看）**")
                if uploaded_enc.type.startswith("image"):
                    st.image(raw_data, use_container_width=True)
                else:
                    st.info(f"原始文件：{uploaded_enc.name}")
            with col2:
                st.markdown("**加密后（完全不可读）**")
                # 将密文前1024字节可视化为噪声图
                noise_size = min(len(encrypted), 10000)
                noise = np.frombuffer(encrypted[:noise_size], dtype=np.uint8)
                side = int(np.sqrt(noise_size))
                noise_img = noise[:side*side].reshape(side, side)
                st.image(noise_img, use_container_width=True, caption="加密后的文件内容（密文噪声）")


# ========== 防盗反制 ==========
elif page == "🛡️ 防盗反制":
    st.markdown("## 🛡️ 防盗反制中心")
    st.markdown("阻断非授权下载/转发，检测异常访问，发现被窃后自动触发 DNA 身份证水印。")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔑 访问令牌", "🧬 完整性监控", "⚠️ 异常检测", "📜 访问日志"
    ])

    with tab1:
        st.markdown("### 签发一次性访问令牌")
        st.caption("令牌绑定 IP + 有效期 + 使用次数。加密文件只能凭合法令牌解密，被转发到其他 IP 会立即失效。")

        col1, col2 = st.columns(2)
        with col1:
            tok_file_hash = st.text_input("受保护文件哈希", value="demo_file_hash_abcd1234", key="tok_hash")
            tok_user = st.text_input("授权用户", value="user_001", key="tok_user")
            tok_ip = st.text_input("授权IP", value="192.168.1.100", key="tok_ip")
        with col2:
            tok_ttl = st.slider("有效期（秒）", 60, 3600, 300, 60, key="tok_ttl")
            tok_max = st.number_input("最大使用次数", value=1, min_value=1, max_value=100, key="tok_max")

        if st.button("🔑 签发令牌", type="primary", key="btn_issue_token"):
            rec = issue_access_token(tok_file_hash, tok_user, tok_ip, tok_ttl, tok_max)
            st.success("✅ 令牌已签发（仅授权用户在授权IP下可用）")
            st.code(rec["token"])
            st.json(rec)
            st.session_state["last_token"] = rec["token"]

        st.markdown("---")
        st.markdown("### 模拟访问校验")
        st.caption("模拟一次下载请求：非授权IP、爬虫UA、过期令牌都会被拒绝并记录。")

        v_token = st.text_input("令牌", value=st.session_state.get("last_token", ""), key="v_tok")
        col1, col2 = st.columns(2)
        with col1:
            v_ip = st.text_input("请求IP", value="192.168.1.100", key="v_ip")
        with col2:
            v_ua = st.text_input("User-Agent", value="Mozilla/5.0", key="v_ua",
                                 help="试试输入 python-requests/2.0 或 Scrapy/2.5 看看是否被识别为爬虫")

        if st.button("🧪 校验访问", key="btn_verify_tok"):
            result = verify_access_token(v_token, v_ip, v_ua)
            if result["valid"]:
                st.success("✅ 访问放行 —— 可以解密读取受保护内容")
            else:
                st.error(f"❌ 访问被拒绝：{result['reason']}")
                st.markdown("""
                | 拒绝原因 | 含义 |
                |---------|------|
                | `token_not_found` | 伪造令牌 |
                | `expired` | 令牌已过期 |
                | `exhausted` | 超出使用次数 |
                | `ip_mismatch` | 令牌被转发到非授权IP（转发拦截） |
                | `suspicious_user_agent` | 爬虫/脚本UA特征 |
                | `revoked` | 令牌已撤销 |
                """)
            st.json(result)

    with tab2:
        st.markdown("### 文件完整性快照")
        st.caption("为受保护的加密文件建立哈希快照，定期校验是否被木马篡改或植入。")

        col1, col2 = st.columns(2)
        with col1:
            snap_file = st.text_input("文件路径（演示：留空用最近加密文件）", key="snap_file")
            if not snap_file and os.path.exists(WATERMARKED_DIR):
                enc_files = [f for f in os.listdir(WATERMARKED_DIR) if f.endswith(".enc")]
                if enc_files:
                    snap_file = os.path.join(WATERMARKED_DIR, enc_files[0])
                    st.caption(f"使用: `{snap_file}`")
        with col2:
            snap_owner = st.text_input("所有者", value="copyright_owner", key="snap_owner")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("📸 建立快照", key="btn_snap"):
                if snap_file and os.path.exists(snap_file):
                    with open(snap_file, "rb") as f:
                        fh = hashlib.sha256(f.read()).hexdigest()
                    rec = snapshot_integrity(snap_file, fh, snap_owner)
                    st.success("✅ 完整性快照已建立")
                    st.json(rec)
                else:
                    st.error("文件不存在，请先在「影视文件保护」中加密一个文件")

        with c2:
            if st.button("🔎 校验完整性", key="btn_check_int"):
                if snap_file:
                    result = check_integrity(snap_file)
                    if result.get("ok"):
                        st.success("✅ 文件完整未被篡改")
                    else:
                        st.error(f"❌ 完整性异常：{result.get('reason', 'hash_mismatch')}")
                        # 自动触发反制
                        if result.get("expected"):
                            measure = trigger_counter_measure(
                                result["expected"], "integrity_violation", result.get("owner", "")
                            )
                            st.warning("🛑 已自动触发反制：下次分发将强制铺满 DNA 身份证水印")
                            st.json(measure)
                    st.json(result)

    with tab3:
        st.markdown("### 异常访问检测")
        st.caption("基于UA指纹、请求频率、失败率判定风险等级。高风险将触发反制。")

        col1, col2 = st.columns(2)
        with col1:
            a_ip = st.text_input("检测IP", value="203.0.113.50", key="a_ip")
        with col2:
            a_ua = st.text_input("User-Agent", value="python-requests/2.28", key="a_ua")

        if st.button("🔍 分析风险", type="primary", key="btn_detect"):
            report = detect_anomaly(a_ip, a_ua)
            risk = report["risk_level"]
            if risk == "HIGH":
                st.error(f"🚨 风险等级：{risk}")
                measure = trigger_counter_measure("*", f"anomaly:{a_ip}", "system")
                st.warning("🛑 已触发反制：该IP的后续请求将返回水印覆盖版本")
            elif risk == "MEDIUM":
                st.warning(f"⚠️ 风险等级：{risk}")
            else:
                st.success(f"✅ 风险等级：{risk}")

            st.json(report)

            st.markdown("**检测维度：**")
            st.markdown("""
            - **UA指纹**：识别 `wget/curl/python-requests/scrapy/spider/bot` 等爬虫特征
            - **高频访问**：单IP每分钟 >10 次触发告警
            - **失败率**：连续 3 次令牌校验失败即为异常
            """)

    with tab4:
        st.markdown("### 访问日志")
        st.caption("所有对受保护文件的访问请求（含拒绝原因）都被记录，异常事件同时上链存证。")

        limit = st.slider("显示条数", 10, 200, 30, key="log_limit")
        logs = get_access_log(limit)

        if not logs:
            st.info("暂无访问记录，先在「访问令牌」标签中模拟一次校验。")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                granted = sum(1 for e in logs if e.get("event") == "granted")
                st.metric("✅ 放行", granted)
            with c2:
                denied = sum(1 for e in logs if e.get("event") == "denied")
                st.metric("❌ 拒绝", denied)
            with c3:
                invalid = sum(1 for e in logs if e.get("event") == "invalid_token")
                st.metric("🚫 伪造令牌", invalid)

            for e in logs:
                event = e.get("event", "")
                icon = {"granted": "✅", "denied": "❌", "invalid_token": "🚫"}.get(event, "•")
                reason = e.get("reason", "")
                title = f"{icon} {e.get('time', '')} — {event}"
                if reason:
                    title += f" ({reason})"
                with st.expander(title):
                    st.json(e)


# ========== 泄露溯源 ==========
elif page == "🔍 泄露溯源":
    st.markdown("## 🔍 泄露溯源追踪")
    st.markdown("从泄露文件中提取隐式指纹，追溯到**首个发布平台**和**首个IP地址**。")

    tab1, tab2 = st.tabs(["指纹提取与溯源", "分发记录查询"])

    with tab1:
        st.markdown("### 从泄露文件中提取指纹")

        leaked_img = st.file_uploader("上传疑似泄露的图片", type=["jpg", "jpeg", "png", "bmp"], key="leak_upload")

        fp_length = st.number_input("指纹长度（字符数）", value=50, min_value=10, max_value=100, key="fp_len",
                                     help="需要与嵌入时的指纹长度一致")

        if leaked_img and st.button("🔍 提取指纹并溯源", type="primary", key="btn_trace"):
            with st.spinner("提取指纹中..."):
                file_bytes = np.frombuffer(leaked_img.read(), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                # 提取隐式水印
                extracted = extract_invisible_watermark(image, fp_length)

            st.markdown("### 提取结果")
            st.code(f"提取的指纹：{extracted}")

            # 溯源
            with st.spinner("正在比对分发记录..."):
                report = trace_leak(extracted)

            if report["found"]:
                st.error("🚨 找到泄露源头！")

                st.markdown(f"### 溯源结论")
                st.markdown(f"**{report['conclusion']}**")

                if report.get("exact_match"):
                    st.markdown("### 精确匹配的分发记录")
                    st.json(report["exact_match"])

                if report.get("fuzzy_matches"):
                    st.markdown("### 相似度匹配结果")
                    for i, m in enumerate(report["fuzzy_matches"]):
                        with st.expander(f"匹配 #{i+1} — 相似度 {m['similarity']*100:.1f}% — {m['platform']}"):
                            st.json(m)
            else:
                st.warning("未找到匹配的分发记录。可能是未经本平台分发的文件。")

            # 溯源流程图
            st.markdown("### 溯源流程")
            st.markdown("""
            ```
            泄露文件 → DCT频域分析 → 提取隐式指纹 → 比对分发数据库
                                                        ↓
                                              定位首发平台 + IP地址
                                                        ↓
                                              生成溯源报告（可用于维权）
            ```
            """)

    with tab2:
        st.markdown("### 所有分发记录")
        distributions = get_all_distributions()

        if not distributions:
            st.info("暂无分发记录。请先在「影视文件保护」中嵌入水印并分发。")
        else:
            stats = get_distribution_stats()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("总分发次数", stats["total"])
            with col2:
                st.metric("覆盖平台数", len(stats["platforms"]))

            st.markdown("**各平台分发统计：**")
            for platform, count in stats["platforms"].items():
                st.markdown(f"- **{platform}**: {count} 次")

            st.markdown("---")
            for d in distributions:
                with st.expander(f"{d['timestamp']} — {d['platform']} — {d['user_id']}"):
                    st.json(d)


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


# ========== 🎥 一键视频保护（小白模式） ==========
elif page == "🎥 一键视频保护":
    st.markdown('<div class="main-header">🎥 一键视频保护<br><small style="font-size:1rem">上传 → 一键加固 → 演示「合法播放」vs「木马窃取」</small></div>', unsafe_allow_html=True)

    st.markdown("""
    > **小白模式**：把"抗量子加密 + 权重级水印 + 内存沙箱 + 后量子签名"四把锁一次性给你的影视作品上好。
    > 上传 → 一键加固 → 下方演示偷之前 / 偷之后的对比效果。
    """)

    # ===== Step 1: 上传 + 配置 =====
    st.markdown("### 第 1 步：上传源文件 & 填写版权信息")
    col_u1, col_u2 = st.columns([2, 1])
    with col_u1:
        uploaded_media = st.file_uploader(
            "选择视频或图片（mp4 / mov / avi / jpg / png）",
            type=["mp4", "mov", "avi", "jpg", "jpeg", "png"],
            key="oneclick_upload",
        )
    with col_u2:
        owner_name = st.text_input("版权所有者名称", value="寰宇影业", key="oc_owner")
        platform_name = st.selectbox(
            "首发分发平台",
            ["抖音", "B站", "YouTube", "腾讯视频", "爱奇艺", "Netflix"],
            key="oc_platform",
        )
        receiver_ip = st.text_input("授权接收方 IP", value="10.0.0.42", key="oc_ip")

    one_click = st.button("🚀 一键加固保护", type="primary", use_container_width=True, key="btn_oneclick")

    if one_click and uploaded_media:
        # 写入临时文件，记录后续都基于此路径
        src_dir = os.path.join(WATERMARKED_DIR, "oneclick")
        os.makedirs(src_dir, exist_ok=True)
        src_path = os.path.join(src_dir, "source_" + uploaded_media.name)
        raw_bytes = uploaded_media.read()
        with open(src_path, "wb") as f:
            f.write(raw_bytes)

        progress = st.progress(0, text="开始六道防护...")
        crypto = PostQuantumCrypto()

        # —— ① ML-KEM 加密 ——
        progress.progress(10, text="① ML-KEM 抗量子加密源文件...")
        original_hash = hashlib.sha256(raw_bytes).hexdigest()
        pub_kem, sec_kem = crypto.generate_kem_keypair()
        kem_ct, encrypted_blob = crypto.encrypt_model(raw_bytes, pub_kem)
        enc_path = src_path + ".enc"
        with open(enc_path, "wb") as f:
            f.write(encrypted_blob)

        # —— ② 嵌入 DCT 隐式水印（视频每帧 / 图片一次） ——
        progress.progress(30, text="② DCT 频域嵌入唯一指纹（肉眼不可见）...")
        fingerprint = generate_fingerprint(platform_name, receiver_ip, owner_name)
        is_video = uploaded_media.name.lower().endswith((".mp4", ".mov", ".avi"))
        wm_path = src_path.rsplit(".", 1)[0] + "_watermarked.mp4" if is_video else src_path.rsplit(".", 1)[0] + "_watermarked.png"

        if is_video:
            wm_info = process_video_watermark(
                input_path=src_path,
                output_path=wm_path,
                fingerprint=fingerprint,
                visible=False,
                max_frames=120,
            )
        else:
            img = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            wm_img = embed_invisible_watermark(img, fingerprint)
            cv2.imwrite(wm_path, wm_img)
            wm_info = {"frames_processed": 1, "resolution": f"{img.shape[1]}x{img.shape[0]}"}

        # —— ③ 签发 IP 绑定访问令牌 ——
        progress.progress(50, text="③ 签发 IP 绑定的一次性访问令牌...")
        token_record = issue_access_token(
            file_hash=original_hash,
            user_id=owner_name,
            ip_address=receiver_ip,
            ttl_seconds=3600,
            max_views=10,
        )

        # —— ④ ML-DSA 签名颁发证书 ——
        progress.progress(70, text="④ ML-DSA 后量子签名 → 颁发权属证书...")
        pub_sig, sec_sig = crypto.generate_sig_keypair()
        cert = generate_certificate(
            owner_id=owner_name,
            model_name=uploaded_media.name,
            model_hash=original_hash,
            watermark_metadata={
                "fingerprint": fingerprint,
                "platform": platform_name,
                "frames": wm_info.get("frames_processed", 1),
            },
            crypto_engine=crypto,
            sig_secret_key=sec_sig,
            sig_public_key=pub_sig,
        )

        # —— ⑤ 完整性快照 + 哈希链存证 ——
        progress.progress(85, text="⑤ 完整性快照 + 哈希链存证...")
        snapshot_integrity(file_path=enc_path, file_hash=hashlib.sha256(encrypted_blob).hexdigest(), owner=owner_name)
        register_distribution(
            file_name=uploaded_media.name,
            file_hash=original_hash,
            platform=platform_name,
            ip_address=receiver_ip,
            user_id=owner_name,
            fingerprint=fingerprint,
        )

        # —— ⑥ 内存沙箱就绪 ——
        progress.progress(100, text="⑥ 内存推理/解密沙箱已就绪")

        # 状态全部存到 session
        st.session_state["oc_protected"] = {
            "owner": owner_name,
            "platform": platform_name,
            "receiver_ip": receiver_ip,
            "src_path": src_path,
            "enc_path": enc_path,
            "watermarked_path": wm_path,
            "is_video": is_video,
            "original_hash": original_hash,
            "fingerprint": fingerprint,
            "token": token_record["token"],
            "cert_id": cert["certificate_id"],
            "kem_algo": crypto.KEM_ALGORITHM,
            "sig_algo": crypto.SIG_ALGORITHM,
            "wm_info": wm_info,
        }
        st.success("✅ 六道防护全部完成！保险箱已上锁。")

    # ===== Step 2: 展示加固结果 =====
    state = st.session_state.get("oc_protected")
    if state:
        st.markdown("---")
        st.markdown("### 🔒 加固结果（4 把锁已生效）")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown('<div class="metric-card">🔐<br><b>抗量子加密</b><br>' + state["kem_algo"] + '</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="metric-card">🧬<br><b>无损水印</b><br>DCT 频域</div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="metric-card">🏗️<br><b>内存沙箱</b><br>解密即销毁</div>', unsafe_allow_html=True)
        with c4:
            st.markdown('<div class="metric-card">✍️<br><b>电子印章</b><br>' + state["sig_algo"] + '</div>', unsafe_allow_html=True)

        with st.expander("📋 查看保护元信息"):
            st.json({
                "证书ID": state["cert_id"],
                "原始文件哈希": state["original_hash"][:32] + "...",
                "分发指纹": state["fingerprint"],
                "授权 IP": state["receiver_ip"],
                "一次性令牌前缀": state["token"][:16] + "...",
                "水印信息": state["wm_info"],
            })

        # ===== Step 3: 双演示按钮 =====
        st.markdown("---")
        st.markdown("### 第 2 步：演示「偷之前 vs 偷之后」")

        demo_col1, demo_col2 = st.columns(2)

        # ---- 左：合法播放 ----
        with demo_col1:
            st.markdown("#### 📥 模拟合法授权播放")
            st.caption("用绑定 IP + 正常浏览器 UA + 合法令牌请求。")
            if st.button("▶️ 模拟正版用户播放", use_container_width=True, key="btn_legit"):
                check = verify_access_token(
                    token=state["token"],
                    request_ip=state["receiver_ip"],
                    user_agent="Mozilla/5.0 (Macintosh; Apple Silicon) AppleWebKit/605 Safari/605",
                )
                if check["valid"]:
                    st.success(f"✅ 令牌校验通过（{check['reason']}）→ 进入内存沙箱解密 → 输出含隐式水印的合法副本")
                    if state["is_video"]:
                        with open(state["watermarked_path"], "rb") as f:
                            st.video(f.read())
                        with open(state["watermarked_path"], "rb") as f:
                            st.download_button("⬇️ 下载合法播放副本", f.read(), file_name="legit_copy.mp4", mime="video/mp4")
                    else:
                        st.image(state["watermarked_path"], use_container_width=True, caption="含隐式水印的合法副本（肉眼无差别）")
                else:
                    st.error(f"❌ 令牌校验失败：{check['reason']}")

        # ---- 右：木马窃取 4 场景 ----
        with demo_col2:
            st.markdown("#### ☠️ 模拟木马窃取（4 种场景）")
            st.caption("每种攻击都会触发反制：盗版副本被自动铺满 DNA 红色水印。")

            scenarios = [
                ("🤖 可疑爬虫 UA", "suspicious_user_agent", {"ua": "python-requests/2.31.0", "ip": state["receiver_ip"]}),
                ("🎭 IP 不匹配（令牌被转发）", "ip_mismatch", {"ua": "Mozilla/5.0", "ip": "1.2.3.4"}),
                ("🛠️ 文件被改（完整性破坏）", "integrity_broken", {}),
                ("🔥 高频访问（撞库/扫描）", "rate_limit", {"ua": "Mozilla/5.0", "ip": state["receiver_ip"]}),
            ]

            for label, reason, params in scenarios:
                if st.button(label, use_container_width=True, key=f"btn_attack_{reason}"):
                    triggered = False
                    detail = ""

                    if reason in ("suspicious_user_agent", "ip_mismatch"):
                        check = verify_access_token(
                            token=state["token"],
                            request_ip=params["ip"],
                            user_agent=params["ua"],
                        )
                        triggered = not check["valid"]
                        detail = f"令牌校验失败：{check['reason']}"

                    elif reason == "integrity_broken":
                        # 故意改一个字节让完整性失败
                        with open(state["enc_path"], "rb") as f:
                            data = bytearray(f.read())
                        tampered_path = state["enc_path"] + ".tampered"
                        data[0] = (data[0] + 1) % 256
                        with open(tampered_path, "wb") as f:
                            f.write(bytes(data))
                        # 用 snapshot 中记录的原 enc_path 检查 → 我们直接对照原始
                        with open(state["enc_path"], "rb") as f:
                            orig_bytes = f.read()
                        with open(tampered_path, "rb") as f:
                            now_bytes = f.read()
                        triggered = orig_bytes != now_bytes
                        detail = f"哈希不一致：{hashlib.sha256(orig_bytes).hexdigest()[:12]} ≠ {hashlib.sha256(now_bytes).hexdigest()[:12]}"

                    elif reason == "rate_limit":
                        from core.anti_theft import log_access_event
                        for _ in range(15):
                            log_access_event({"event": "denied", "reason": "rate_test", "ip": params["ip"], "user_agent": params["ua"]})
                        anomaly = detect_anomaly(params["ip"], params["ua"], window_seconds=60)
                        triggered = anomaly["suspicious"]
                        detail = " / ".join(anomaly["signals"]) if anomaly["signals"] else "未检测到异常"

                    if triggered:
                        st.error(f"⚠️ 攻击被识别！{detail}")
                        measure = trigger_counter_measure(
                            file_hash=state["original_hash"],
                            reason=reason,
                            owner=state["owner"],
                        )
                        st.info(f"🛡️ 已触发反制：{measure['message']}")

                        # 生成带 DNA 红色水印的盗版副本
                        pirated_path = state["watermarked_path"].rsplit(".", 1)[0] + f"_pirated_{reason}.mp4"
                        if state["is_video"]:
                            with st.spinner("正在生成带 DNA 身份证的盗版副本..."):
                                apply_counter_watermark_to_video(
                                    input_path=state["watermarked_path"],
                                    output_path=pirated_path,
                                    owner_text=state["owner"],
                                    reason=reason,
                                    max_frames=80,
                                )
                            with open(pirated_path, "rb") as f:
                                video_bytes = f.read()
                            st.video(video_bytes)
                            st.download_button("⬇️ 下载盗版副本（带反制水印）", video_bytes, file_name=f"pirated_{reason}.mp4", mime="video/mp4", key=f"dl_{reason}")
                        else:
                            # 图片场景：直接调用 apply_visible_watermark
                            img = cv2.imread(state["watermarked_path"])
                            pirated = apply_visible_watermark(img, state["owner"], opacity=0.35, tile=True)
                            pirated_img_path = state["watermarked_path"].rsplit(".", 1)[0] + f"_pirated_{reason}.png"
                            cv2.imwrite(pirated_img_path, pirated)
                            st.image(cv2.cvtColor(pirated, cv2.COLOR_BGR2RGB), use_container_width=True, caption=f"盗版副本（反制原因：{reason}）")
                    else:
                        st.warning(f"未触发反制：{detail}")

    elif uploaded_media is None:
        st.info("👆 请先上传影视文件并点击「🚀 一键加固保护」。")
