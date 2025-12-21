"""
環境變數管理模組
統一管理所有環境變數的讀取
"""
import os
from pathlib import Path
from typing import Optional

# 嘗試載入 python-dotenv
try:
    from dotenv import load_dotenv
    _DOTENV_AVAILABLE = True
except ImportError:
    _DOTENV_AVAILABLE = False


def _find_env_file() -> Optional[Path]:
    """尋找 .env 檔案（從當前目錄向上搜尋）"""
    current = Path.cwd()
    
    for parent in [current] + list(current.parents):
        env_file = parent / ".env"
        if env_file.exists():
            return env_file
    
    # 也檢查專案根目錄
    project_root = Path(__file__).parent
    env_file = project_root / ".env"
    if env_file.exists():
        return env_file
    
    return None


def load_env():
    """載入環境變數"""
    if _DOTENV_AVAILABLE:
        env_file = _find_env_file()
        if env_file:
            load_dotenv(env_file)
            print(f"[ENV] Loaded from: {env_file}")
        else:
            print("[ENV] No .env file found, using system environment variables")
    else:
        print("[ENV] python-dotenv not installed, using system environment variables only")


# 自動載入
load_env()


def get_env(key: str, default: str = "") -> str:
    """取得環境變數"""
    return os.getenv(key, default)


def get_env_int(key: str, default: int = 0) -> int:
    """取得整數環境變數"""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def get_env_bool(key: str, default: bool = False) -> bool:
    """取得布林環境變數"""
    value = os.getenv(key, "").lower()
    if value in ("true", "1", "yes", "on"):
        return True
    if value in ("false", "0", "no", "off"):
        return False
    return default


# ============================================
# 伺服器設定
# ============================================

# Game Server
SERVER_NAME = get_env("SERVER_NAME", "Agar.io AI Lab (Auto-Scale)")
SERVER_HOST = get_env("SERVER_HOST", "localhost")
SERVER_PORT = get_env_int("SERVER_PORT", 8765)
MAX_PLAYERS = get_env_int("MAX_PLAYERS", 50)

# Master Server
MASTER_HOST = get_env("MASTER_HOST", "localhost")
MASTER_PORT = get_env_int("MASTER_PORT", 8080)
MASTER_URL = get_env("MASTER_URL", f"http://{MASTER_HOST}:{MASTER_PORT}")

# Health Check
HEALTH_CHECK_INTERVAL = get_env_int("HEALTH_CHECK_INTERVAL", 30)
OFFLINE_TIMEOUT = get_env_int("OFFLINE_TIMEOUT", 300)

# Debug
DEBUG = get_env_bool("DEBUG", False)
