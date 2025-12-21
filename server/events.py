"""
遊戲事件類型定義
"""
from enum import Enum


class GameEvent(str, Enum):
    """遊戲事件類型"""
    EAT_FOOD = "eat_food"
    EAT_EJECTED = "eat_ejected"
    EAT_VIRUS = "eat_virus"
    EAT_PLAYER_CELL = "eat_player"
    SPLIT_CELL = "split"
    EJECT_MASS = "eject"
    MERGE_CELLS = "merge"
    VIRUS_EXPLODE = "virus_explode"
    VIRUS_SPLIT = "virus_split"
