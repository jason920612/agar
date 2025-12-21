"""
AI Bot 模組
包含遺傳演算法與 Bot 行為邏輯
"""
import copy
import math
import random
import time
from typing import Dict, List, Optional, Any

from .config import config
from .entities import Player, Cell


# Bot 名稱庫
BOT_NAMES = [
    "Taiwan", "USA", "China", "Japan", "Korea", "Russia", "Germany", "France", "UK", "Italy",
    "Canada", "Australia", "Brazil", "India", "Vietnam", "Thailand", "Singapore", "Malaysia",
    "Tokyo", "New York", "London", "Paris", "Beijing", "Shanghai", "Taipei", "Seoul", "Moscow",
    "Hong Kong", "Berlin", "Rome", "Washington", "California", "Texas", "Florida",
    "Trump", "Biden", "Obama", "Putin", "Xi Jinping", "Merkel", "Macron", "Zelensky",
    "Kim Jong-un", "Modi", "Trudeau", "Thatcher", "Churchill", "Kennedy", "Lincoln",
    "Elon Musk", "Zuckerberg", "Bill Gates", "Jobs"
]


class EvolutionManager:
    """遺傳演算法管理器"""
    
    def __init__(self):
        self.gene_pool: List[tuple] = []
        self.generation_count = 0
        self.best_score = 0
        self.base_genes = {
            'w_food': (5000, 20000),
            'w_hunt': (1000000, 5000000),
            'w_flee': (-8000000, -2000000),
            'w_virus': (-20000, 50000),
            'split_dist': (200, 700),
            'split_aggr': (1.1, 1.5)
        }
    
    def create_random_genes(self) -> Dict[str, Any]:
        """創建隨機基因"""
        genes = {}
        for key, (min_v, max_v) in self.base_genes.items():
            genes[key] = random.uniform(min_v, max_v)
        genes['generation'] = 1
        return genes
    
    def record_genome(self, bot: 'Bot'):
        """記錄基因組"""
        score = bot.max_mass_achieved + (time.time() - bot.birth_time) * 2
        if score > self.best_score:
            self.best_score = score
        
        self.gene_pool.append((score, copy.deepcopy(bot.genes)))
        self.gene_pool.sort(key=lambda x: x[0], reverse=True)
        self.gene_pool = self.gene_pool[:15]  # 只保留前 15 個
    
    def get_next_generation_genes(self) -> Dict[str, Any]:
        """取得下一代基因"""
        if not self.gene_pool or random.random() < 0.2:
            return self.create_random_genes()
        
        parent_genes = random.choice(self.gene_pool)[1]
        child_genes = copy.deepcopy(parent_genes)
        
        # 隨機突變
        for key in self.base_genes:
            if random.random() < 0.3:
                child_genes[key] *= random.uniform(0.8, 1.2)
        
        child_genes['generation'] = parent_genes.get('generation', 1) + 1
        return child_genes


# 全域演化管理器
evo_manager = EvolutionManager()


class Bot(Player):
    """AI Bot 玩家"""
    
    def __init__(self, pid: int, genes: Optional[Dict] = None, 
                 map_size: tuple = (6000, 6000)):
        bot_name = random.choice(BOT_NAMES)
        super().__init__(None, pid, bot_name, ip="BOT-AI", 
                        spectate=False, map_size=map_size)
        self.color = "#%06x" % random.randint(0, 0xFFFFFF)
        self.genes = genes if genes else evo_manager.create_random_genes()
    
    def decide(self, world) -> Optional[str]:
        """決定下一步動作"""
        if self.is_dead or not self.cells:
            return None
        
        current_mass = self.total_mass
        if current_mass > self.max_mass_achieved:
            self.max_mass_achieved = current_mass
        
        if not self.cells:
            return None
        
        my_largest = max(self.cells, key=lambda c: c.mass)
        mx, my = my_largest.x, my_largest.y
        view_dist = 800 + my_largest.radius * 5
        
        target_x, target_y = 0.0, 0.0
        
        # 食物權重
        w_food = self.genes['w_food']
        gx, gy = int(mx // config.GRID_SIZE), int(my // config.GRID_SIZE)
        search_grids = [(gx, gy), (gx+1, gy), (gx-1, gy), (gx, gy+1), (gx, gy-1)]
        
        food_vec_x, food_vec_y = 0.0, 0.0
        for g in search_grids:
            if g in world.food_grid:
                for f in world.food_grid[g]:
                    dx = f['x'] - mx
                    dy = f['y'] - my
                    d2 = dx*dx + dy*dy
                    if d2 < view_dist**2:
                        weight = w_food / (d2 + 1)
                        food_vec_x += dx * weight
                        food_vec_y += dy * weight
        
        target_x += food_vec_x
        target_y += food_vec_y
        
        action_intent = None
        
        # 獵殺 / 逃跑權重
        w_hunt = self.genes['w_hunt']
        w_flee = self.genes['w_flee']
        split_dist = self.genes['split_dist']
        split_aggr = self.genes['split_aggr']
        
        for pid, p in world.players.items():
            if p.id == self.id or p.is_dead or p.is_spectator:
                continue
            
            for enemy_cell in p.cells:
                dx = enemy_cell.x - mx
                dy = enemy_cell.y - my
                dist = math.sqrt(dx**2 + dy**2)
                
                if dist > view_dist or dist <= 0.1:
                    continue
                
                if enemy_cell.mass > my_largest.mass * 1.15:
                    # 逃跑
                    weight = w_flee / (dist + 1)
                    target_x += (dx / dist) * weight
                    target_y += (dy / dist) * weight
                elif enemy_cell.mass * split_aggr < my_largest.mass:
                    # 獵殺
                    weight = w_hunt / (dist + 1)
                    target_x += (dx / dist) * weight
                    target_y += (dy / dist) * weight
                    
                    if my_largest.mass > 50 and len(self.cells) < config.MAX_CELLS:
                        if dist < split_dist:
                            action_intent = 'split'
        
        # 病毒權重
        w_virus = self.genes['w_virus']
        for v in world.viruses:
            dx = v.x - mx
            dy = v.y - my
            dist = math.sqrt(dx**2 + dy**2)
            
            if dist > view_dist or dist <= 0.1:
                continue
            
            if my_largest.mass > v.mass * 1.15:
                if len(self.cells) >= config.MAX_CELLS:
                    # 已經分裂到極限，可以吃病毒
                    target_x += (dx / dist) * 50000
                    target_y += (dy / dist) * 50000
                else:
                    # 躲避病毒
                    if dist < my_largest.radius + 100:
                        weight = w_virus / (dist + 1)
                        target_x += (dx / dist) * weight
                        target_y += (dy / dist) * weight
        
        # 計算最終方向
        final_len = math.sqrt(target_x**2 + target_y**2)
        if final_len > 0:
            self.mouse_x = mx + (target_x / final_len) * 500
            self.mouse_y = my + (target_y / final_len) * 500
        else:
            map_w, map_h = self._map_size
            self.mouse_x = random.randint(0, map_w)
            self.mouse_y = random.randint(0, map_h)
        
        return action_intent
