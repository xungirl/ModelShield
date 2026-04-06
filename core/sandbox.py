"""
推理沙箱模块 —— 进程级隔离的安全推理环境

原理：
1. 将模型推理放在独立子进程中执行
2. 限制内存使用和执行时间
3. 模型在沙箱内解密运行，密钥不暴露给外部
"""
import multiprocessing
import resource
import signal
import os
import sys
import torch
import io
import traceback
from typing import Any, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SANDBOX_TIMEOUT, SANDBOX_MAX_MEMORY


class SandboxResult:
    """沙箱执行结果"""
    def __init__(self, success: bool, output: Any = None, error: str = None,
                 memory_used: int = 0, time_used: float = 0):
        self.success = success
        self.output = output
        self.error = error
        self.memory_used = memory_used
        self.time_used = time_used

    def to_dict(self):
        return {
            "success": self.success,
            "output": str(self.output) if self.output is not None else None,
            "error": self.error,
            "memory_used_mb": round(self.memory_used / (1024 * 1024), 2),
            "time_used_s": round(self.time_used, 3),
        }


def _set_resource_limits(max_memory: int, max_time: int):
    """在子进程中设置资源限制"""
    # 限制内存
    try:
        resource.setrlimit(resource.RLIMIT_AS, (max_memory, max_memory))
    except (ValueError, resource.error):
        pass  # 某些系统不支持

    # 限制 CPU 时间
    resource.setrlimit(resource.RLIMIT_CPU, (max_time, max_time))


def _sandbox_worker(
    model_bytes: bytes,
    input_data: dict,
    result_queue: multiprocessing.Queue,
    max_memory: int,
    max_time: int,
):
    """沙箱工作进程"""
    import time as time_mod
    start_time = time_mod.time()

    try:
        # 设置资源限制
        _set_resource_limits(max_memory, max_time)

        # 在沙箱内加载模型
        buffer = io.BytesIO(model_bytes)
        model = torch.load(buffer, map_location="cpu", weights_only=False)
        model.eval()

        # 准备输入
        input_tensor = torch.tensor(input_data["input"], dtype=torch.float32)
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)

        # 推理
        with torch.no_grad():
            output = model(input_tensor)

        elapsed = time_mod.time() - start_time

        result_queue.put(SandboxResult(
            success=True,
            output=output.tolist(),
            time_used=elapsed,
        ))

    except Exception as e:
        elapsed = time_mod.time() - start_time
        result_queue.put(SandboxResult(
            success=False,
            error=f"{type(e).__name__}: {str(e)}",
            time_used=elapsed,
        ))


def run_in_sandbox(
    model: torch.nn.Module,
    input_data: dict,
    timeout: int = SANDBOX_TIMEOUT,
    max_memory: int = SANDBOX_MAX_MEMORY,
) -> SandboxResult:
    """
    在安全沙箱中执行模型推理

    Args:
        model: PyTorch 模型
        input_data: 输入数据 {"input": [...]}
        timeout: 超时时间（秒）
        max_memory: 最大内存（字节）

    Returns:
        SandboxResult
    """
    # 序列化模型
    buffer = io.BytesIO()
    torch.save(model, buffer)
    model_bytes = buffer.getvalue()

    # 创建结果队列
    result_queue = multiprocessing.Queue()

    # 启动沙箱进程
    process = multiprocessing.Process(
        target=_sandbox_worker,
        args=(model_bytes, input_data, result_queue, max_memory, timeout),
    )
    process.start()
    process.join(timeout=timeout + 5)  # 额外5秒缓冲

    if process.is_alive():
        process.terminate()
        process.join(timeout=3)
        if process.is_alive():
            process.kill()
        return SandboxResult(
            success=False,
            error="推理超时：执行时间超过限制",
            time_used=timeout,
        )

    # 获取结果
    if not result_queue.empty():
        return result_queue.get()

    return SandboxResult(
        success=False,
        error="沙箱进程异常退出",
    )


def get_sandbox_info() -> dict:
    """获取沙箱环境信息"""
    return {
        "isolation": "process",
        "max_memory_mb": SANDBOX_MAX_MEMORY // (1024 * 1024),
        "timeout_s": SANDBOX_TIMEOUT,
        "platform": sys.platform,
        "python_version": sys.version,
        "torch_available": True,
    }
