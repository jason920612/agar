"""
Master Server 設定
"""
import sys
from pathlib import Path

# 加入專案根目錄到路徑
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from env import (
    MASTER_HOST, MASTER_PORT,
    HEALTH_CHECK_INTERVAL, OFFLINE_TIMEOUT as ENV_OFFLINE_TIMEOUT,
    DEBUG
)

# 伺服器設定 (從環境變數讀取)
CHECK_INTERVAL = HEALTH_CHECK_INTERVAL
OFFLINE_TIMEOUT = ENV_OFFLINE_TIMEOUT
LISTEN_HOST = MASTER_HOST
LISTEN_PORT = MASTER_PORT
