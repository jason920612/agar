"""
遊戲伺服器設定模組
"""
import json
import sys
from pathlib import Path

# 加入專案根目錄到路徑
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from env import (
    SERVER_NAME, SERVER_HOST, SERVER_PORT,
    MAX_PLAYERS, MASTER_URL, DEBUG
)

# 設定檔路徑（相對於專案根目錄）
CONFIG_PATH = PROJECT_ROOT / "game_config.json"

# 預設設定
DEFAULT_CONFIG = {
    "map_width": 6000,
    "map_height": 6000,
    "player_start_mass": 20,
    "virus_count": 30,
    "virus_start_mass": 100,
    "virus_max_mass": 180,
    "food_max_count": 1200,
    "food_min_mass": 1,
    "food_max_mass": 2,
    "mass_decay_rate": 0.003,
    "merge_time_factor": 30,
    "merge_attraction_force": 0.15,
    "max_cell_mass": 22600,
    "dynamic_scaling_enabled": True,
    "scaling_player_step": 5,
    "scaling_size_percent": 0.2,
    "scaling_resource_percent": 0.2
}


class GameConfig:
    """遊戲設定管理類"""
    
    def __init__(self):
        self._config = self._load_config()
        
        # 伺服器設定 (從環境變數讀取)
        self.SERVER_NAME = SERVER_NAME
        self.SERVER_HOST = SERVER_HOST
        self.SERVER_PORT = SERVER_PORT
        self.MAX_PLAYERS = MAX_PLAYERS
        self.MASTER_URL = MASTER_URL
        self.DEBUG = DEBUG
        
        # 物理常數
        self.TICK_RATE = 20
        self.TICK_LEN = 1 / self.TICK_RATE
        self.MAX_CELLS = 16
        self.SPLIT_IMPULSE = 780
        self.EJECT_IMPULSE = 550
        self.FRICTION = 0.90
        self.VIRUS_SHOT_IMPULSE = 850
        self.GRID_SIZE = 300
    
    def _load_config(self) -> dict:
        """載入設定檔"""
        if CONFIG_PATH.exists():
            try:
                with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    # 合併預設值（確保新增的設定項有預設值）
                    return {**DEFAULT_CONFIG, **loaded}
            except Exception as e:
                print(f"Error loading config: {e}, using defaults.")
                return DEFAULT_CONFIG.copy()
        else:
            print("Config file not found, using defaults.")
            return DEFAULT_CONFIG.copy()
    
    def reload(self):
        """重新載入設定"""
        self._config = self._load_config()
        print("Configuration reloaded successfully.")
    
    # --- 遊戲參數屬性 ---
    @property
    def map_width(self) -> int:
        return self._config['map_width']
    
    @property
    def map_height(self) -> int:
        return self._config['map_height']
    
    @property
    def player_start_mass(self) -> int:
        return self._config.get('player_start_mass', 20)
    
    @property
    def virus_count(self) -> int:
        return self._config['virus_count']
    
    @property
    def virus_start_mass(self) -> int:
        return self._config['virus_start_mass']
    
    @property
    def virus_max_mass(self) -> int:
        return self._config['virus_max_mass']
    
    @property
    def food_max_count(self) -> int:
        return self._config.get('food_max_count', 1200)
    
    @property
    def food_min_mass(self) -> int:
        return self._config.get('food_min_mass', 1)
    
    @property
    def food_max_mass(self) -> int:
        return self._config.get('food_max_mass', 2)
    
    @property
    def mass_decay_rate(self) -> float:
        return self._config['mass_decay_rate']
    
    @property
    def merge_time_factor(self) -> float:
        return self._config.get('merge_time_factor', 30)
    
    @property
    def merge_attraction_force(self) -> float:
        return self._config.get('merge_attraction_force', 0.15)
    
    @property
    def max_cell_mass(self) -> int:
        return self._config.get('max_cell_mass', 22600)
    
    @property
    def dynamic_scaling_enabled(self) -> bool:
        return self._config.get('dynamic_scaling_enabled', False)
    
    @property
    def scaling_player_step(self) -> int:
        return self._config.get('scaling_player_step', 5)
    
    @property
    def scaling_size_percent(self) -> float:
        return self._config.get('scaling_size_percent', 0.2)
    
    @property
    def scaling_resource_percent(self) -> float:
        return self._config.get('scaling_resource_percent', 0.2)


# 全域設定實例
config = GameConfig()
