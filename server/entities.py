"""
遊戲實體類別模組
包含: GameObject, Cell, Player, EjectedMass, Virus
"""
import math
import random
import time
from typing import List, Optional, Tuple

from .config import config


def mass_to_radius(mass: float) -> float:
    """將質量轉換為半徑"""
    return 6 * math.sqrt(max(0, mass))


def clamp(n: float, min_val: float, max_val: float) -> float:
    """限制數值在範圍內"""
    return max(min(max_val, n), min_val)


class GameObject:
    """遊戲物件基礎類別"""
    
    def __init__(self, x: float, y: float, mass: float, color: str):
        self.x = x
        self.y = y
        self.mass = mass
        self.color = color
        self.id = random.randint(0, 100_000_000)
    
    @property
    def radius(self) -> float:
        return mass_to_radius(self.mass)


class EjectedMass(GameObject):
    """噴射的質量球"""
    
    def __init__(self, x: float, y: float, angle: float, color: str, 
                 parent_id: int, team_id: Optional[int]):
        super().__init__(x, y, 16, color)
        self.vx = math.cos(angle) * config.EJECT_IMPULSE
        self.vy = math.sin(angle) * config.EJECT_IMPULSE
        self.parent_id = parent_id
        self.team_id = team_id
        self.birth_time = time.time()
    
    def move(self, map_width: int, map_height: int):
        """更新位置"""
        self.x = clamp(self.x + self.vx * config.TICK_LEN, 0, map_width)
        self.y = clamp(self.y + self.vy * config.TICK_LEN, 0, map_height)
        self.vx *= config.FRICTION
        self.vy *= config.FRICTION


class Virus(GameObject):
    """病毒"""
    
    def __init__(self, x: float, y: float, mass: Optional[float] = None, 
                 angle: float = 0, velocity: float = 0):
        if mass is None:
            mass = config.virus_start_mass
        super().__init__(x, y, mass, "#33ff33")
        self.vx = math.cos(angle) * velocity
        self.vy = math.sin(angle) * velocity
    
    def move(self, map_width: int, map_height: int):
        """更新位置"""
        if self.vx != 0 or self.vy != 0:
            self.x = clamp(self.x + self.vx * config.TICK_LEN, 0, map_width)
            self.y = clamp(self.y + self.vy * config.TICK_LEN, 0, map_height)
            self.vx *= config.FRICTION
            self.vy *= config.FRICTION
            if abs(self.vx) < 1 and abs(self.vy) < 1:
                self.vx = self.vy = 0


class Cell(GameObject):
    """玩家細胞"""
    
    def __init__(self, x: float, y: float, mass: float, color: str):
        super().__init__(x, y, mass, color)
        self.boost_x = 0.0
        self.boost_y = 0.0
        self.set_recombine_cooldown()
    
    def set_recombine_cooldown(self):
        """設定合併冷卻時間"""
        base_factor = config.merge_time_factor
        start_mass = max(config.player_start_mass, 2)
        current_mass = max(self.mass, 2)
        log_ratio = math.log(current_mass) / math.log(start_mass)
        recombine_seconds = base_factor * log_ratio
        self.recombine_time = time.time() + recombine_seconds
    
    def apply_force(self, fx: float, fy: float):
        """施加推力"""
        self.boost_x += fx
        self.boost_y += fy
    
    def move(self, tx: float, ty: float, map_width: int, map_height: int):
        """向目標移動"""
        if math.isnan(tx) or math.isnan(ty):
            return
        
        dx, dy = tx - self.x, ty - self.y
        dist = math.sqrt(dx**2 + dy**2)
        safe_mass = max(self.mass, 1)
        base_speed = 300 * (safe_mass ** -0.2)
        
        if dist > 0:
            speed = min(dist * 5, base_speed)
            self.x += (dx / dist) * speed * config.TICK_LEN
            self.y += (dy / dist) * speed * config.TICK_LEN
        
        # 套用推力
        self.x += self.boost_x * config.TICK_LEN
        self.y += self.boost_y * config.TICK_LEN
        self.boost_x *= config.FRICTION
        self.boost_y *= config.FRICTION
        
        # 限制在地圖範圍內
        self.x = clamp(self.x, 0, map_width)
        self.y = clamp(self.y, 0, map_height)
    
    def decay(self):
        """質量衰減"""
        if self.mass > config.player_start_mass:
            self.mass -= self.mass * config.mass_decay_rate * config.TICK_LEN
            if self.mass < config.player_start_mass:
                self.mass = config.player_start_mass


class Player:
    """玩家類別"""
    
    def __init__(self, ws, pid: int, name: str, ip: str = "Unknown", 
                 spectate: bool = False, map_size: Tuple[int, int] = (6000, 6000)):
        self.ws = ws
        self.id = pid
        self.name = name
        self.ip = ip
        self.cells: List[Cell] = []
        self.color = "#%06x" % random.randint(0, 0xFFFFFF)
        self.mouse_x = map_size[0] / 2
        self.mouse_y = map_size[1] / 2
        self.is_dead = True
        self.is_spectator = spectate
        self.team_id = None
        self.birth_time = time.time()
        self.max_mass_achieved = 0
        self._map_size = map_size
        
        if not spectate:
            self.spawn()
    
    @property
    def total_mass(self) -> float:
        return sum(c.mass for c in self.cells)
    
    @property
    def center(self) -> Tuple[float, float]:
        if self.is_spectator:
            return (self.mouse_x, self.mouse_y)
        if not self.cells:
            return (self._map_size[0] / 2, self._map_size[1] / 2)
        return (
            sum(c.x for c in self.cells) / len(self.cells),
            sum(c.y for c in self.cells) / len(self.cells)
        )
    
    def spawn(self):
        """重生"""
        if self.is_spectator:
            return
        
        start_mass = config.player_start_mass
        spawn_x = random.randint(100, self._map_size[0] - 100)
        spawn_y = random.randint(100, self._map_size[1] - 100)
        self.cells = [Cell(spawn_x, spawn_y, start_mass, self.color)]
        self.is_dead = False
        self.birth_time = time.time()
        self.max_mass_achieved = start_mass
    
    def update_map_size(self, width: int, height: int):
        """更新地圖大小參考"""
        self._map_size = (width, height)
