import asyncio
import json
import math
import random
import time
import websockets
import sys
import aiohttp 
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
import copy

# --- 讀取設定檔邏輯 ---
DEFAULT_CONFIG = {
    "map_width": 6000,
    "map_height": 6000,
    "player_start_mass": 20,
    "virus_count": 30,
    "virus_start_mass": 100,
    "virus_max_mass": 180,
    "food_max_count": 1200,
    "food_min_mass": 5,
    "food_max_mass": 10,
    "mass_decay_rate": 0.006,
    "merge_time_factor": 30,
    "merge_attraction_force": 0.15,
    "max_cell_mass": 22600,          # 新增預設值
    "dynamic_scaling_enabled": True, # 新增預設值
    "scaling_player_step": 5,        # 新增預設值
    "scaling_size_percent": 0.2,     # 新增預設值
    "scaling_resource_percent": 0.2  # 新增預設值
}

def load_config():
    if os.path.exists('game_config.json'):
        try:
            with open('game_config.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}, using defaults.")
            return DEFAULT_CONFIG
    else:
        print("Config file not found, using defaults.")
        return DEFAULT_CONFIG

GAME_CONFIG = load_config()

# --- 伺服器設定 ---
SERVER_NAME = "Agar.io AI Lab (Auto-Scale)"
SERVER_HOST = "localhost" 
SERVER_PORT = 8765
MAX_PLAYERS = 50
MASTER_URL = "http://localhost:8080" 
MY_URL = f"ws://{SERVER_HOST}:{SERVER_PORT}"

# --- 遊戲參數 (初始值) ---
MAP_WIDTH = GAME_CONFIG['map_width']
MAP_HEIGHT = GAME_CONFIG['map_height']
VIRUS_COUNT = GAME_CONFIG['virus_count']
VIRUS_START_MASS = GAME_CONFIG['virus_start_mass']
VIRUS_MAX_MASS = GAME_CONFIG['virus_max_mass']
MASS_DECAY_RATE = GAME_CONFIG['mass_decay_rate']
PLAYER_START_MASS = GAME_CONFIG.get('player_start_mass', 20)
MERGE_ATTRACTION = GAME_CONFIG.get('merge_attraction_force', 0.15)
MAX_CELL_MASS = GAME_CONFIG.get('max_cell_mass', 22600)

# --- 動態縮放基準 ---
BASE_MAP_WIDTH = MAP_WIDTH
BASE_MAP_HEIGHT = MAP_HEIGHT
BASE_VIRUS_COUNT = VIRUS_COUNT
BASE_FOOD_COUNT = GAME_CONFIG.get('food_max_count', 1200)

# --- 物理與常數 ---
TICK_RATE = 20
TICK_LEN = 1 / TICK_RATE
MAX_CELLS = 16
SPLIT_IMPULSE = 780    
EJECT_IMPULSE = 550
FRICTION = 0.90
VIRUS_SHOT_IMPULSE = 850
GRID_SIZE = 300 

# --- BOT 名稱庫 ---
BOT_NAMES = [
    "Taiwan", "USA", "China", "Japan", "Korea", "Russia", "Germany", "France", "UK", "Italy",
    "Canada", "Australia", "Brazil", "India", "Vietnam", "Thailand", "Singapore", "Malaysia",
    "Tokyo", "New York", "London", "Paris", "Beijing", "Shanghai", "Taipei", "Seoul", "Moscow",
    "Hong Kong", "Berlin", "Rome", "Washington", "California", "Texas", "Florida",
    "Trump", "Biden", "Obama", "Putin", "Xi Jinping", "Merkel", "Macron", "Zelensky", 
    "Kim Jong-un", "Modi", "Trudeau", "Thatcher", "Churchill", "Kennedy", "Lincoln",
    "Elon Musk", "Zuckerberg", "Bill Gates", "Jobs"
]

# --- 輔助函數 ---
def mass_to_radius(mass): return 6 * math.sqrt(max(0, mass))
def clamp(n, minn, maxn): return max(min(maxn, n), minn)

# --- 事件類型 ---
class GameEvent:
    EAT_FOOD = "eat_food"
    EAT_EJECTED = "eat_ejected"
    EAT_VIRUS = "eat_virus"
    EAT_PLAYER_CELL = "eat_player"
    SPLIT_CELL = "split"
    EJECT_MASS = "eject"
    MERGE_CELLS = "merge"
    VIRUS_EXPLODE = "virus_explode"
    VIRUS_SPLIT = "virus_split" 

# --- 遺傳演算法管理器 (無變更) ---
class EvolutionManager:
    def __init__(self):
        self.gene_pool = [] 
        self.generation_count = 0
        self.best_score = 0
        self.base_genes = {
            'w_food': (5000, 20000), 'w_hunt': (1000000, 5000000), 'w_flee': (-8000000, -2000000),
            'w_virus': (-20000, 50000), 'split_dist': (200, 700), 'split_aggr': (1.1, 1.5)
        }
    def create_random_genes(self):
        genes = {}
        for key, (min_v, max_v) in self.base_genes.items(): genes[key] = random.uniform(min_v, max_v)
        genes['generation'] = 1
        return genes
    def record_genome(self, bot):
        score = bot.max_mass_achieved + (time.time() - bot.birth_time) * 2
        if score > self.best_score: self.best_score = score
        self.gene_pool.append((score, copy.deepcopy(bot.genes)))
        self.gene_pool.sort(key=lambda x: x[0], reverse=True)
        self.gene_pool = self.gene_pool[:15]
    def get_next_generation_genes(self):
        if not self.gene_pool or random.random() < 0.2: return self.create_random_genes()
        parent_genes = random.choice(self.gene_pool)[1]
        child_genes = copy.deepcopy(parent_genes)
        for key in self.base_genes:
            if random.random() < 0.3: child_genes[key] *= random.uniform(0.8, 1.2)
        child_genes['generation'] = parent_genes.get('generation', 1) + 1
        return child_genes
evo_manager = EvolutionManager()

# --- 類別定義 ---
class GameObject:
    def __init__(self, x, y, mass, color):
        self.x, self.y = x, y
        self.mass = mass
        self.color = color
        self.id = random.randint(0, 100000000)
    @property
    def radius(self): return mass_to_radius(self.mass)

class EjectedMass(GameObject):
    def __init__(self, x, y, angle, color, parent_id, team_id):
        super().__init__(x, y, 16, color)
        self.vx = math.cos(angle) * EJECT_IMPULSE
        self.vy = math.sin(angle) * EJECT_IMPULSE
        self.parent_id = parent_id
        self.team_id = team_id
        self.birth_time = time.time()
    def move(self):
        self.x = clamp(self.x + self.vx * TICK_LEN, 0, MAP_WIDTH)
        self.y = clamp(self.y + self.vy * TICK_LEN, 0, MAP_HEIGHT)
        self.vx *= FRICTION
        self.vy *= FRICTION

class Virus(GameObject):
    def __init__(self, x, y, mass=None, angle=0, velocity=0):
        if mass is None: mass = VIRUS_START_MASS
        super().__init__(x, y, mass, "#33ff33")
        self.vx = math.cos(angle) * velocity
        self.vy = math.sin(angle) * velocity
    def move(self):
        if self.vx != 0 or self.vy != 0:
            self.x = clamp(self.x + self.vx * TICK_LEN, 0, MAP_WIDTH)
            self.y = clamp(self.y + self.vy * TICK_LEN, 0, MAP_HEIGHT)
            self.vx *= FRICTION
            self.vy *= FRICTION
            if abs(self.vx) < 1 and abs(self.vy) < 1: self.vx = self.vy = 0

class Cell(GameObject):
    def __init__(self, x, y, mass, color):
        super().__init__(x, y, mass, color)
        self.boost_x = 0
        self.boost_y = 0
        self.set_recombine_cooldown()
    def set_recombine_cooldown(self):
        base_factor = GAME_CONFIG.get('merge_time_factor', 30)
        start_mass = GAME_CONFIG.get('player_start_mass', 20)
        if start_mass < 2: start_mass = 2 
        current_mass = max(self.mass, 2)
        log_ratio = math.log(current_mass) / math.log(start_mass)
        recombine_seconds = base_factor * log_ratio
        self.recombine_time = time.time() + recombine_seconds
    def apply_force(self, fx, fy):
        self.boost_x += fx; self.boost_y += fy
    def move(self, tx, ty):
        if math.isnan(tx) or math.isnan(ty): return
        dx, dy = tx - self.x, ty - self.y
        dist = math.sqrt(dx**2 + dy**2)
        safe_mass = max(self.mass, 1)
        base_speed = 300 * (safe_mass ** -0.2) 
        if dist > 0:
            speed = min(dist * 5, base_speed)
            self.x += (dx/dist) * speed * TICK_LEN
            self.y += (dy/dist) * speed * TICK_LEN
        self.x += self.boost_x * TICK_LEN
        self.y += self.boost_y * TICK_LEN
        self.boost_x *= FRICTION
        self.boost_y *= FRICTION
    def decay(self):
        if self.mass > PLAYER_START_MASS:
            self.mass -= self.mass * MASS_DECAY_RATE * TICK_LEN
            if self.mass < PLAYER_START_MASS: self.mass = PLAYER_START_MASS

class Player:
    def __init__(self, ws, pid, name, ip="Unknown", spectate=False):
        self.ws = ws
        self.id = pid
        self.name = name
        self.ip = ip
        self.cells = []
        self.color = "#%06x" % random.randint(0, 0xFFFFFF)
        self.mouse_x, self.mouse_y = MAP_WIDTH/2, MAP_HEIGHT/2
        self.is_dead = True
        self.is_spectator = spectate
        self.team_id = None 
        self.birth_time = time.time()
        self.max_mass_achieved = 0
        if not spectate: self.spawn()
    @property
    def total_mass(self): return sum([c.mass for c in self.cells])
    @property
    def center(self):
        if self.is_spectator: return (self.mouse_x, self.mouse_y)
        if not self.cells: return (MAP_WIDTH/2, MAP_HEIGHT/2)
        return (sum(c.x for c in self.cells)/len(self.cells), sum(c.y for c in self.cells)/len(self.cells))
    def spawn(self):
        if self.is_spectator: return
        start_m = PLAYER_START_MASS
        self.cells = [Cell(random.randint(100, MAP_WIDTH-100), random.randint(100, MAP_HEIGHT-100), start_m, self.color)]
        self.is_dead = False
        self.birth_time = time.time()
        self.max_mass_achieved = start_m

class Bot(Player):
    def __init__(self, pid, genes=None):
        bot_name = random.choice(BOT_NAMES)
        super().__init__(None, pid, bot_name, ip="BOT-AI", spectate=False)
        self.color = "#%06x" % random.randint(0, 0xFFFFFF)
        self.genes = genes if genes else evo_manager.create_random_genes()
    def decide(self, world):
        if self.is_dead or not self.cells: return None
        current_mass = self.total_mass
        if current_mass > self.max_mass_achieved: self.max_mass_achieved = current_mass
        if not self.cells: return None
        my_largest = max(self.cells, key=lambda c: c.mass)
        mx, my = my_largest.x, my_largest.y
        view_dist = 800 + my_largest.radius * 5
        target_x, target_y = 0, 0
        w_food = self.genes['w_food']
        gx, gy = int(mx // GRID_SIZE), int(my // GRID_SIZE)
        search_grids = [(gx, gy), (gx+1, gy), (gx-1, gy), (gx, gy+1), (gx, gy-1)]
        food_vec_x, food_vec_y = 0, 0
        for g in search_grids:
            if g in world.food_grid:
                for f in world.food_grid[g]:
                    dx = f['x'] - mx; dy = f['y'] - my
                    d2 = dx*dx + dy*dy
                    if d2 < view_dist**2:
                        weight = w_food / (d2 + 1)
                        food_vec_x += dx * weight
                        food_vec_y += dy * weight
        target_x += food_vec_x; target_y += food_vec_y
        action_intent = None
        w_hunt = self.genes['w_hunt']
        w_flee = self.genes['w_flee']
        split_dist = self.genes['split_dist']
        split_aggr = self.genes['split_aggr']
        for pid, p in world.players.items():
            if p.id == self.id or p.is_dead or p.is_spectator: continue
            for enemy_cell in p.cells:
                dx = enemy_cell.x - mx; dy = enemy_cell.y - my
                dist = math.sqrt(dx**2 + dy**2)
                if dist > view_dist or dist <= 0.1: continue
                if enemy_cell.mass > my_largest.mass * 1.15:
                    weight = w_flee / (dist + 1)
                    target_x += (dx / dist) * weight
                    target_y += (dy / dist) * weight
                elif enemy_cell.mass * split_aggr < my_largest.mass:
                    weight = w_hunt / (dist + 1)
                    target_x += (dx / dist) * weight
                    target_y += (dy / dist) * weight
                    if my_largest.mass > 50 and len(self.cells) < MAX_CELLS:
                        if dist < split_dist: action_intent = 'split'
        w_virus = self.genes['w_virus']
        for v in world.viruses:
            dx = v.x - mx; dy = v.y - my
            dist = math.sqrt(dx**2 + dy**2)
            if dist > view_dist or dist <= 0.1: continue
            if my_largest.mass > v.mass * 1.15:
                if len(self.cells) >= MAX_CELLS:
                     target_x += (dx/dist) * 50000
                     target_y += (dy/dist) * 50000
                else:
                    if dist < my_largest.radius + 100:
                        weight = w_virus / (dist + 1)
                        target_x += (dx/dist) * weight
                        target_y += (dy/dist) * weight
        final_len = math.sqrt(target_x**2 + target_y**2)
        if final_len > 0:
            self.mouse_x = mx + (target_x / final_len) * 500
            self.mouse_y = my + (target_y / final_len) * 500
        else:
            self.mouse_x = random.randint(0, MAP_WIDTH)
            self.mouse_y = random.randint(0, MAP_HEIGHT)
        return action_intent

class GameWorld:
    def __init__(self):
        self.players = {}
        self.food = []
        self.food_grid = {} 
        self.viruses = []
        self.ejected_mass = []
        self.event_queue = [] 
        self.max_food = BASE_FOOD_COUNT
        self.target_virus_count = BASE_VIRUS_COUNT
        
        self.generate_food(int(self.max_food * 0.6))
        self.generate_viruses(self.target_virus_count)
        
        # 動態縮放狀態
        self.last_scaling_check = 0
        self.map_needs_sync = False # 用於通知廣播

    def add_food_to_grid(self, f):
        gx, gy = int(f['x']//GRID_SIZE), int(f['y']//GRID_SIZE)
        if (gx, gy) not in self.food_grid: self.food_grid[(gx, gy)] = []
        self.food_grid[(gx, gy)].append(f)

    def remove_food_from_grid(self, f):
        gx, gy = int(f['x']//GRID_SIZE), int(f['y']//GRID_SIZE)
        if (gx, gy) in self.food_grid:
            try: self.food_grid[(gx, gy)].remove(f)
            except: pass
            if not self.food_grid[(gx, gy)]: del self.food_grid[(gx, gy)]

    def generate_food(self, amount):
        min_m = GAME_CONFIG.get('food_min_mass', 5)
        max_m = GAME_CONFIG.get('food_max_mass', 10)
        for _ in range(amount):
            f = {
                'id': random.randint(0, 1000000000), 
                'x': random.randint(0,MAP_WIDTH), 
                'y': random.randint(0,MAP_HEIGHT), 
                'color': "#%06x"%random.randint(0,0xFFFFFF), 
                'mass': random.randint(min_m, max_m)
            }
            self.food.append(f)
            self.add_food_to_grid(f)
            
    def generate_viruses(self, amount):
        for _ in range(amount):
            self.viruses.append(Virus(random.randint(100, MAP_WIDTH-100), random.randint(100, MAP_HEIGHT-100)))

    def check_dynamic_scaling(self):
        """計算並應用動態地圖與資源縮放"""
        global MAP_WIDTH, MAP_HEIGHT, VIRUS_COUNT
        if not GAME_CONFIG.get('dynamic_scaling_enabled', False): return

        player_count = len([p for p in self.players.values() if not p.is_spectator])
        step = GAME_CONFIG.get('scaling_player_step', 5)
        size_pct = GAME_CONFIG.get('scaling_size_percent', 0.2)
        res_pct = GAME_CONFIG.get('scaling_resource_percent', 0.2)

        multiplier = player_count // step
        
        # 計算新參數
        new_width = int(BASE_MAP_WIDTH * (1 + multiplier * size_pct))
        new_height = int(BASE_MAP_HEIGHT * (1 + multiplier * size_pct))
        new_max_food = int(BASE_FOOD_COUNT * (1 + multiplier * res_pct))
        new_virus_count = int(BASE_VIRUS_COUNT * (1 + multiplier * res_pct))

        # 如果發生變化則更新
        if new_width != MAP_WIDTH or new_height != MAP_HEIGHT:
            print(f"[Scale] Players: {player_count}, Size: {new_width}x{new_height}, Food: {new_max_food}, Virus: {new_virus_count}")
            MAP_WIDTH = new_width
            MAP_HEIGHT = new_height
            self.max_food = new_max_food
            self.target_virus_count = new_virus_count
            VIRUS_COUNT = new_virus_count
            self.map_needs_sync = True # 標記需要廣播

    def enforce_max_mass(self):
        """強制分裂過大的細胞"""
        for p in self.players.values():
            if p.is_dead or p.is_spectator: continue
            
            # 使用新的列表儲存分裂出的細胞，避免在迭代中修改列表
            cells_to_add = []
            
            for c in p.cells:
                if c.mass >= MAX_CELL_MASS:
                    # 執行強制分裂邏輯
                    split_mass = c.mass / 2
                    c.mass = split_mass
                    c.set_recombine_cooldown() # 重置合併冷卻

                    # 計算分裂方向 (向滑鼠方向)
                    dx, dy = p.mouse_x - c.x, p.mouse_y - c.y
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist == 0: dx, dy = 1, 0
                    else: dx, dy = dx/dist, dy/dist

                    # 創建新細胞
                    nc = Cell(c.x + dx * c.radius, c.y + dy * c.radius, split_mass, c.color)
                    
                    # 施加推力
                    c.apply_force(-dx * 100, -dy * 100) # 母細胞稍微後退
                    nc.apply_force(dx * SPLIT_IMPULSE, dy * SPLIT_IMPULSE) # 子細胞射出
                    
                    cells_to_add.append(nc)

            # 如果有新細胞，加入玩家列表
            if cells_to_add:
                p.cells.extend(cells_to_add)
                # 如果超過最大細胞數量限制，這裡不強制刪除，因為這是懲罰性分裂
                # 但如果需要嚴格遵守 16 上限，可以在這裡做修剪，暫時允許超過以便懲罰效果顯著

    def update(self):
        now = time.time()
        
        # 週期性檢查地圖縮放 (每 5 秒)
        if now - self.last_scaling_check > 5:
            self.check_dynamic_scaling()
            self.last_scaling_check = now

        # 強制分裂檢查
        self.enforce_max_mass()

        for e in self.ejected_mass: e.move()
        for v in self.viruses: v.move()
        
        # --- BOT Evolution & Update Logic ---
        for p in list(self.players.values()): 
            if isinstance(p, Bot):
                if p.is_dead:
                    evo_manager.record_genome(p)
                    new_genes = evo_manager.get_next_generation_genes()
                    del self.players[p.id]
                    global pid_counter
                    new_id = pid_counter; pid_counter += 1
                    new_bot = Bot(new_id, genes=new_genes)
                    new_bot.spawn()
                    self.players[new_id] = new_bot
                    continue
                else:
                    action = p.decide(self)
                    if action == 'split': self.event_queue.append({'type': GameEvent.SPLIT_CELL, 'player': p})
                    elif action == 'eject': self.event_queue.append({'type': GameEvent.EJECT_MASS, 'player': p})

        active_players = [p for p in self.players.values() if not p.is_dead]

        for p in active_players:
            if len(p.cells) > 1:
                for i in range(len(p.cells)):
                    for j in range(i + 1, len(p.cells)):
                        c1 = p.cells[i]
                        c2 = p.cells[j]
                        if now > c1.recombine_time and now > c2.recombine_time:
                            dx = c2.x - c1.x; dy = c2.y - c1.y
                            dist = math.sqrt(dx**2 + dy**2)
                            if dist > 0:
                                attraction_factor = MERGE_ATTRACTION 
                                safe_mass1 = max(c1.mass, 1); safe_mass2 = max(c2.mass, 1)
                                base_speed_c1 = 300 * (safe_mass1 ** -0.2)
                                base_speed_c2 = 300 * (safe_mass2 ** -0.2)
                                pull_x = (dx / dist) * TICK_LEN; pull_y = (dy / dist) * TICK_LEN
                                c1.x += pull_x * base_speed_c1 * attraction_factor; c1.y += pull_y * base_speed_c1 * attraction_factor
                                c2.x -= pull_x * base_speed_c2 * attraction_factor; c2.y -= pull_y * base_speed_c2 * attraction_factor

        for p in active_players:
            for cell in p.cells:
                cell.move(p.mouse_x, p.mouse_y)
                cell.x = clamp(cell.x, 0, MAP_WIDTH)
                cell.y = clamp(cell.y, 0, MAP_HEIGHT)
                cell.decay()

        for p in active_players:
            for cell in p.cells:
                min_gx = int((cell.x - cell.radius)//GRID_SIZE)
                max_gx = int((cell.x + cell.radius)//GRID_SIZE)
                min_gy = int((cell.y - cell.radius)//GRID_SIZE)
                max_gy = int((cell.y + cell.radius)//GRID_SIZE)
                for gx in range(min_gx, max_gx+1):
                    for gy in range(min_gy, max_gy+1):
                        if (gx, gy) in self.food_grid:
                            for f in self.food_grid[(gx, gy)]:
                                if (cell.x-f['x'])**2 + (cell.y-f['y'])**2 < cell.radius**2:
                                    self.event_queue.append({'type': GameEvent.EAT_FOOD, 'cell': cell, 'food': f})

                for e in self.ejected_mass:
                    if e.parent_id == p.id and now - e.birth_time < 0.2: continue
                    if (cell.x-e.x)**2 + (cell.y-e.y)**2 < cell.radius**2:
                        self.event_queue.append({'type': GameEvent.EAT_EJECTED, 'cell': cell, 'ejected': e})

                for v in self.viruses:
                    if cell.mass > v.mass * 1.1 and (cell.x-v.x)**2 + (cell.y-v.y)**2 < cell.radius**2:
                        self.event_queue.append({'type': GameEvent.EAT_VIRUS, 'player': p, 'cell_idx': p.cells.index(cell), 'virus': v})

        # --- 細胞合併邏輯 ---
        for p1 in active_players:
            for i in range(len(p1.cells)):
                for j in range(i+1, len(p1.cells)):
                    c1, c2 = p1.cells[i], p1.cells[j]
                    dist = math.sqrt((c1.x-c2.x)**2 + (c1.y-c2.y)**2)
                    if c1.mass > c2.mass: big_cell, small_cell = c1, c2
                    else: big_cell, small_cell = c2, c1
                    radius_sum = c1.radius + c2.radius
                    if dist < radius_sum:
                        can_merge = (now > c1.recombine_time) and (now > c2.recombine_time)
                        if can_merge and dist < radius_sum * 0.65:
                            self.event_queue.append({'type': GameEvent.MERGE_CELLS, 'player': p1, 'idx1': i, 'idx2': j})
                        elif not can_merge:
                             pen = radius_sum - dist
                             if dist == 0: 
                                 rand_ang = random.uniform(0, math.pi*2)
                                 dx, dy = math.cos(rand_ang), math.sin(rand_ang)
                             else: dx, dy = (c1.x-c2.x)/dist, (c1.y-c2.y)/dist
                             f = 0.5
                             c1.x += dx * pen * f; c1.y += dy * pen * f
                             c2.x -= dx * pen * f; c2.y -= dy * pen * f
            
            for p2 in active_players:
                if p1.id == p2.id: continue
                for c1 in p1.cells:
                    for c2 in p2.cells:
                        if c1.mass > c2.mass * 1.25 and math.sqrt((c1.x-c2.x)**2+(c1.y-c2.y)**2) < c1.radius:
                             self.event_queue.append({'type': GameEvent.EAT_PLAYER_CELL, 'predator': c1, 'prey': c2, 'prey_p': p2})
        
        for v in self.viruses:
             for e in self.ejected_mass:
                 if (v.x-e.x)**2 + (v.y-e.y)**2 < (v.radius+e.radius)**2:
                     self.event_queue.append({'type': GameEvent.VIRUS_SPLIT, 'virus': v, 'ejected': e})

        self.process_events()

        if len(self.food) < self.max_food: self.generate_food(10)
        if len(self.viruses) < self.target_virus_count: self.generate_viruses(1)

    def process_events(self):
        removed_food = set()
        removed_ejected = set()
        removed_viruses = set()
        
        for e in self.event_queue:
            t = e['type']
            if t == GameEvent.EAT_FOOD:
                f = e['food']
                if f['id'] not in removed_food:
                    e['cell'].mass += f['mass']
                    removed_food.add(f['id'])
                    self.food.remove(f)
                    self.remove_food_from_grid(f)
            elif t == GameEvent.EAT_EJECTED:
                ej = e['ejected']
                if ej.id not in removed_ejected:
                    e['cell'].mass += ej.mass
                    removed_ejected.add(ej.id)
                    if ej in self.ejected_mass: self.ejected_mass.remove(ej)
            elif t == GameEvent.EAT_PLAYER_CELL:
                prey = e['prey']
                if prey.mass > 0: 
                    e['predator'].mass += prey.mass
                    prey.mass = 0 
            elif t == GameEvent.SPLIT_CELL:
                p = e['player']
                if len(p.cells) < MAX_CELLS:
                    new_cells = []
                    for c in p.cells:
                        if c.mass >= 36 and len(p.cells)+len(new_cells) < MAX_CELLS:
                            split_mass = c.mass / 2
                            c.mass = split_mass
                            dx, dy = p.mouse_x - c.x, p.mouse_y - c.y
                            ang = math.atan2(dy, dx)
                            c.apply_force(-math.cos(ang)*200, -math.sin(ang)*200)
                            nc = Cell(c.x + math.cos(ang)*c.radius, c.y + math.sin(ang)*c.radius, split_mass, c.color)
                            nc.apply_force(math.cos(ang)*SPLIT_IMPULSE, math.sin(ang)*SPLIT_IMPULSE)
                            new_cells.append(nc)
                    p.cells.extend(new_cells)
            elif t == GameEvent.EJECT_MASS:
                p = e['player']
                for c in p.cells:
                    if c.mass > 36:
                        c.mass -= 16
                        dx, dy = p.mouse_x - c.x, p.mouse_y - c.y
                        ang = math.atan2(dy, dx)
                        c.apply_force(-math.cos(ang)*100, -math.sin(ang)*100)
                        ej_x = c.x + math.cos(ang) * c.radius
                        ej_y = c.y + math.sin(ang) * c.radius
                        self.ejected_mass.append(EjectedMass(ej_x, ej_y, ang, c.color, p.id, p.team_id))
            elif t == GameEvent.EAT_VIRUS:
                v = e['virus']
                if v.id not in removed_viruses:
                    p = e['player']
                    c = p.cells[e['cell_idx']]
                    c.mass += v.mass
                    removed_viruses.add(v.id)
                    if v in self.viruses: self.viruses.remove(v)
                    if len(p.cells) < MAX_CELLS:
                         pieces = min(MAX_CELLS - len(p.cells), 7)
                         if pieces > 0:
                            pmass = c.mass / (pieces + 1); c.mass = pmass; c.set_recombine_cooldown() 
                            for i in range(pieces):
                                ang = (i/pieces) * math.pi * 2 + random.uniform(-0.5, 0.5)
                                nc = Cell(c.x + math.cos(ang)*c.radius*0.8, c.y + math.sin(ang)*c.radius*0.8, pmass, c.color)
                                nc.apply_force(math.cos(ang)*SPLIT_IMPULSE, math.sin(ang)*SPLIT_IMPULSE)
                                p.cells.append(nc)
            elif t == GameEvent.MERGE_CELLS:
                p = e['player']
                try:
                    c1, c2 = p.cells[e['idx1']], p.cells[e['idx2']]
                    if c1.mass > 0 and c2.mass > 0:
                        c1.mass += c2.mass
                        c1.x = (c1.x * c1.mass + c2.x * c2.mass) / (c1.mass + c2.mass)
                        c1.y = (c1.y * c1.mass + c2.y * c2.mass) / (c1.mass + c2.mass)
                        c2.mass = 0 
                except: pass
            elif t == GameEvent.VIRUS_SPLIT:
                v = e['virus']
                ej = e['ejected']
                if ej.id not in removed_ejected and v.id not in removed_viruses:
                    v.mass += ej.mass
                    removed_ejected.add(ej.id)
                    if ej in self.ejected_mass: self.ejected_mass.remove(ej)
                    if v.mass >= VIRUS_MAX_MASS:
                        v.mass = VIRUS_START_MASS
                        shoot_ang = math.atan2(ej.vy, ej.vx)
                        self.viruses.append(Virus(v.x + math.cos(shoot_ang)*v.radius*2.5, v.y + math.sin(shoot_ang)*v.radius*2.5, VIRUS_START_MASS, shoot_ang, VIRUS_SHOT_IMPULSE))

        for p in self.players.values():
            p.cells = [c for c in p.cells if c.mass > 0]
            if not p.cells and not p.is_spectator: p.is_dead = True
            
        self.event_queue.clear()

    def get_view_state(self, player):
        cx, cy = player.center
        v_rad = 2500 
        v_rad_sq = v_rad**2

        visible_players = []
        for p in self.players.values():
            if p.is_dead or p.is_spectator: continue
            p_cells = []
            in_view = False
            for c in p.cells:
                if (c.x-cx)**2 + (c.y-cy)**2 < v_rad_sq:
                    in_view = True
                p_cells.append({'x': int(c.x), 'y': int(c.y), 'm': int(c.mass), 'c': c.color})
            if in_view:
                visible_players.append({'id': p.id, 'name': p.name, 'dead': p.is_dead, 'cells': p_cells})
        
        visible_food = []
        min_gx, max_gx = max(0, int((cx-v_rad)//GRID_SIZE)), min(int(MAP_WIDTH//GRID_SIZE), int((cx+v_rad)//GRID_SIZE))
        min_gy, max_gy = max(0, int((cy-v_rad)//GRID_SIZE)), min(int(MAP_HEIGHT//GRID_SIZE), int((cy+v_rad)//GRID_SIZE))
        
        for gx in range(min_gx, max_gx+1):
            for gy in range(min_gy, max_gy+1):
                if (gx, gy) in self.food_grid:
                    for f in self.food_grid[(gx, gy)]:
                        if (f['x']-cx)**2 + (f['y']-cy)**2 < v_rad_sq:
                            visible_food.append(f)

        visible_viruses = [{'x': int(v.x), 'y': int(v.y), 'm': int(v.mass)} for v in self.viruses if (v.x-cx)**2+(v.y-cy)**2 < v_rad_sq]
        visible_ejected = [{'x': int(e.x), 'y': int(e.y), 'c': e.color} for e in self.ejected_mass if (e.x-cx)**2+(e.y-cy)**2 < v_rad_sq]
        
        lb = []
        for p in sorted([p for p in self.players.values() if not p.is_dead and not p.is_spectator], key=lambda x: x.total_mass, reverse=True)[:10]:
            pcx, pcy = p.center
            lb.append({'id': p.id, 'name': p.name, 'mass': int(p.total_mass), 'x': int(pcx), 'y': int(pcy)})

        return {'players': visible_players, 'food': visible_food, 'viruses': visible_viruses, 'ejected': visible_ejected, 'leaderboard': lb}

def manage_game_commands(cmd):
    global pid_counter, MAP_WIDTH, MAP_HEIGHT
    
    command = cmd[0].lower()

    if command == "reload":
        print("Reloading configuration...")
        try:
            global GAME_CONFIG
            GAME_CONFIG = load_config()
            global VIRUS_COUNT, VIRUS_START_MASS, VIRUS_MAX_MASS, MASS_DECAY_RATE, PLAYER_START_MASS, MERGE_ATTRACTION, MAX_CELL_MASS
            VIRUS_COUNT = GAME_CONFIG['virus_count']
            VIRUS_START_MASS = GAME_CONFIG['virus_start_mass']
            VIRUS_MAX_MASS = GAME_CONFIG['virus_max_mass']
            MASS_DECAY_RATE = GAME_CONFIG['mass_decay_rate']
            PLAYER_START_MASS = GAME_CONFIG.get('player_start_mass', 20)
            MERGE_ATTRACTION = GAME_CONFIG.get('merge_attraction_force', 0.15)
            MAX_CELL_MASS = GAME_CONFIG.get('max_cell_mass', 22600)
            
            if 'world' in globals():
                world.max_food = GAME_CONFIG['food_max_count']
            print("Configuration reloaded successfully.")
        except Exception as e:
            print(f"Error reloading config: {e}")

    elif command == "setsize" and len(cmd) == 3:
        try:
            w, h = int(cmd[1]), int(cmd[2])
            MAP_WIDTH, MAP_HEIGHT = w, h
            world.map_needs_sync = True # 觸發廣播
            print(f"Map size updated to {w}x{h}")
        except ValueError: print("Usage: setsize <width> <height>")

    elif command == "clearfood":
        world.food = []
        world.food_grid = {}
        print("All food cleared.")

    elif command == "foodcfg" and len(cmd) == 3:
        try:
            max_f, rate = int(cmd[1]), int(cmd[2])
            world.max_food = max_f
            print(f"Food Config Updated: Max={max_f}")
        except ValueError: print("Usage: foodcfg <max_amount> <spawn_rate>")

    elif command == "addbot":
        try:
            count = int(cmd[1]) if len(cmd) > 1 else 1
            for _ in range(count):
                bot_id = pid_counter
                pid_counter += 1
                bot = Bot(bot_id, genes=None)
                world.players[bot_id] = bot
                print(f"Bot {bot_id} added.")
        except ValueError: print("Usage: addbot <count>")

    elif command == "removebot":
        try:
            count = int(cmd[1]) if len(cmd) > 1 else 1
            removed = 0
            bot_ids = [pid for pid, p in world.players.items() if isinstance(p, Bot)]
            for i in range(min(count, len(bot_ids))):
                del world.players[bot_ids[i]]
                removed += 1
            print(f"Removed {removed} bots.")
        except ValueError: print("Usage: removebot <count>")

    elif command == "stats":
        print(f"--- Evo Stats ---")
        print(f"Best Score: {int(evo_manager.best_score)}")
        print(f"Gene Pool Size: {len(evo_manager.gene_pool)}")
        if evo_manager.gene_pool:
             best_gene = evo_manager.gene_pool[0][1]
             print(f"Top Gene (Gen {best_gene['generation']}): Hunt={int(best_gene['w_hunt'])}, Flee={int(best_gene['w_flee'])}")

    elif command == "find":
        if len(cmd) < 2:
            print("Usage: find <name_fragment>")
        else:
            target = cmd[1].lower()
            found = False
            print(f"{'ID':<6} {'Name':<15} {'Mass':<8} {'Cells':<6} {'IP Address'}")
            print("-" * 55)
            for p in world.players.values():
                if target in p.name.lower():
                    print(f"{p.id:<6} {p.name[:15]:<15} {int(p.total_mass):<8} {len(p.cells):<6} {p.ip}")
                    found = True
            if not found: print("No matches found.")

    elif command == "addmass":
        if len(cmd) < 3:
            print("Usage: addmass <player_id> <amount>")
        else:
            try:
                target_id = int(cmd[1])
                amount = int(cmd[2])
                p = world.players.get(target_id)
                if p and not p.is_dead and len(p.cells) > 0:
                    per_cell = amount / len(p.cells)
                    for c in p.cells:
                        c.mass += per_cell
                    print(f"Added {amount} mass to {p.name} (ID: {target_id}).")
                else:
                    print("Player not found or is dead.")
            except ValueError: print("Invalid ID or Amount.")

    elif command == "removemass":
        if len(cmd) < 3:
            print("Usage: removemass <player_id> <amount>")
        else:
            try:
                target_id = int(cmd[1])
                amount = int(cmd[2])
                p = world.players.get(target_id)
                if p and not p.is_dead and len(p.cells) > 0:
                    per_cell_loss = amount / len(p.cells)
                    for c in p.cells:
                        c.mass = max(10, c.mass - per_cell_loss) 
                    print(f"Removed {amount} mass from {p.name} (ID: {target_id}).")
                else:
                    print("Player not found or is dead.")
            except ValueError: print("Invalid ID or Amount.")

    elif command == "kill":
        if len(cmd) < 2:
            print("Usage: kill <player_id>")
        else:
            try:
                target_id = int(cmd[1])
                p = world.players.get(target_id)
                if p:
                    p.is_dead = True
                    p.cells = [] 
                    print(f"Killed player {p.name} (ID: {target_id}).")
                else:
                    print("Player not found.")
            except ValueError: print("Invalid ID.")

    elif command == "killbotall":
        count = 0
        for p in list(world.players.values()):
            if isinstance(p, Bot):
                p.is_dead = True
                p.cells = []
                count += 1
        print(f"Killed {count} bots.")

    else:
        print("Commands: reload, setsize, clearfood, foodcfg, addbot, removebot, stats, find, addmass, removemass, kill, killbotall")

async def input_loop():
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(1, "InputThread") as executor:
        while True:
            cmd = await loop.run_in_executor(executor, sys.stdin.readline)
            cmd = cmd.strip().split()
            if not cmd: continue
            manage_game_commands(cmd)

world = GameWorld()
pid_counter = 0

async def handler(ws):
    global pid_counter
    pid = pid_counter; pid_counter += 1
    player = None
    
    remote_ip = "Unknown"
    if ws.remote_address:
        remote_ip = f"{ws.remote_address[0]}:{ws.remote_address[1]}"

    try:
        async for msg in ws:
            data = json.loads(msg)
            if data['type'] == 'ping': await ws.send(json.dumps({'type':'pong', 'server_name':SERVER_NAME, 'players':len(world.players), 'max_players':MAX_PLAYERS}))
            elif data['type'] == 'join':
                player = Player(ws, pid, data.get('name', 'Guest')[:15], ip=remote_ip)
                world.players[pid] = player
                print(f"[Join] ID:{pid} Name:{player.name} IP:{remote_ip}") 
                await ws.send(json.dumps({'type':'init', 'id':pid, 'map':{'w':MAP_WIDTH, 'h':MAP_HEIGHT}}))
            elif data['type'] == 'spectate':
                player = Player(ws, pid, "Spectator", ip=remote_ip, spectate=True)
                world.players[pid] = player
                await ws.send(json.dumps({'type':'init', 'id':pid, 'map':{'w':MAP_WIDTH, 'h':MAP_HEIGHT}}))
            elif player and not player.is_spectator:
                if data['type'] == 'input': 
                    tx, ty = data.get('x'), data.get('y')
                    if tx is not None and ty is not None and not math.isnan(tx) and not math.isnan(ty):
                        player.mouse_x, player.mouse_y = tx, ty
                elif data['type'] == 'split': world.event_queue.append({'type': GameEvent.SPLIT_CELL, 'player': player})
                elif data['type'] == 'eject': world.event_queue.append({'type': GameEvent.EJECT_MASS, 'player': player})
            elif player and player.is_spectator:
                 if data['type'] == 'input':
                    tx, ty = data.get('x'), data.get('y')
                    if tx is not None and ty is not None and not math.isnan(tx) and not math.isnan(ty):
                        player.mouse_x, player.mouse_y = tx, ty
    except: pass
    finally: 
        if pid in world.players: del world.players[pid]

async def game_loop():
    print("Game loop started (Auto-Scale Mode).")
    while True:
        t1 = time.time()
        try:
            world.update()
        except Exception:
            print("!!! CRITICAL ERROR IN GAME LOOP !!!")
            traceback.print_exc()
            await asyncio.sleep(1) # 防止連續錯誤導致 log 爆炸
        
        active_players_snapshot = list(world.players.items())
        
        # --- 廣播地圖大小變更 ---
        if world.map_needs_sync:
            map_data = json.dumps({'type': 'map_update', 'map': {'w': MAP_WIDTH, 'h': MAP_HEIGHT}})
            for pid, p in active_players_snapshot:
                if isinstance(p, Bot): continue
                try: await p.ws.send(map_data)
                except: pass
            world.map_needs_sync = False # 重置標記

        for pid, p in active_players_snapshot:
            if isinstance(p, Bot): continue 
            
            try: 
                if p.is_dead and not p.is_spectator:
                     await p.ws.send(json.dumps({'type': 'death'}))
                else:
                     await p.ws.send(json.dumps({'type':'update', 'data': world.get_view_state(p)}))
            except websockets.exceptions.ConnectionClosed:
                pass 
            except Exception as e:
                print(f"Send Error: {e}") 

        process_time = time.time() - t1
        delay = TICK_LEN - process_time
        await asyncio.sleep(max(0.001, delay))

async def main():
    print(f"Running {SERVER_NAME} on {SERVER_PORT}")
    try:
        async with aiohttp.ClientSession() as s: await s.post(f"{MASTER_URL}/register", json={'url':MY_URL, 'name':SERVER_NAME, 'max_players':MAX_PLAYERS})
    except: pass
    
    server = websockets.serve(handler, SERVER_HOST, SERVER_PORT, ping_interval=None, ping_timeout=None)
    
    await asyncio.gather(server, game_loop(), input_loop())

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: pass