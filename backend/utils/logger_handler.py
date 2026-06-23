import logging
import os
from backend.utils.path_tool import get_abs_path
from datetime import date

# 日志保存的根目录
LOG_ROOT = get_abs_path('logs')

# 创建日志目录
os.makedirs(LOG_ROOT, exist_ok=True)

# 日志格式配置
DEFAULT_LOG_FORMAT = '%(levelname)s - %(asctime)s - %(filename)s:%(lineno)d - %(message)s'

def get_logger(
        name: str = "agent",
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        log_file: str = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 避免重复添加handler，handler即日志出现的地方
    if logger.handlers:
        return logger  
    
    # 添加控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
    logger.addHandler(console_handler)

    # 添加文件handler
    if not log_file:
        log_file = os.path.join(LOG_ROOT, f"{name}-{date.today()}.log")
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
    logger.addHandler(file_handler)

    return logger

logger = get_logger()

if __name__ == '__main__':
    logger.info("信息日志")
    logger.error("错误日志")