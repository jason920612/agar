import asyncio
import json
import math
import random
import time
import websockets
import sys
import aiohttp 
from concurrent.futures import ThreadPoolExecutor
import copy

# --- 伺服器設定 ---
SERVER_NAME = "Agar.io AI Lab (Admin Tools)"
SERVER_HOST = "localhost" 
SERVER_PORT = 8765
MAX_PLAYERS = 50
MASTER_URL = "http://localhost:8080" 
MY_URL = f"ws://{SERVER_HOST}:{SERVER_PORT}"

# --- 遊戲常數 ---
MAP_WIDTH = 6000
MAP_HEIGHT = 6000
TICK_RATE = 20
TICK_LEN = 1 / TICK_RATE
MASS_DECAY_RATE = 0.006 

# --- 物理與平衡參數 ---
BASE_MASS = 20
MAX_CELLS = 16
SPLIT_IMPULSE = 780    
EJECT_IMPULSE = 550
FRICTION = 0.90
VIRUS_START_MASS = 100
VIRUS_MAX_MASS = 180
VIRUS_COUNT = 50 
VIRUS_SHOT_IMPULSE = 850

# --- 視野優化參數 ---
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
def mass_to_radius(mass): return 6 * math.sqrt(mass)
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

# --- 遺傳演算法管理器 ---
class EvolutionManager:
    def __init__(self):
        self.gene_pool = [] 
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

    def create_random_genes(self):
        genes = {}
        for key, (min_v, max_v) in self.base_genes.items():
            genes[key] = random.uniform(min_v, max_v)
        genes['generation'] = 1
        return genes

    def record_genome(self, bot):
        score = bot.max_mass_achieved + (time.time() - bot.birth_time) * 2
        if score > self.best_score:
            self.best_score = score
        self.gene_pool.append((score, copy.deepcopy(bot.genes)))
        self.gene_pool.sort(key=lambda x: x[0], reverse=True)
        self.gene_pool = self.gene_pool[:15]

    def get_next_generation_genes(self):
        if not self.gene_pool or random.random() < 0.2:
            return self.create_random_genes()
        
        parent_genes = random.choice(self.gene_pool)[1]
        child_genes = copy.deepcopy(parent_genes)
        
        for key in self.base_genes:
            if random.random() < 0.3: 
                mutation_factor = random.uniform(0.8, 1.2)
                child_genes[key] *= mutation_factor
        
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
    def __init__(self, x, y, mass=VIRUS_START_MASS, angle=0, velocity=0):
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
        self.recombine_time = time.time() + (30 + (0.02 * self.mass))
    def apply_force(self, fx, fy):
        self.boost_x += fx; self.boost_y += fy
    def move(self, tx, ty):
        if math.isnan(tx) or math.isnan(ty): return
        dx, dy = tx - self.x, ty - self.y
        dist = math.sqrt(dx**2 + dy**2)
        base_speed = 300 * (self.mass ** -0.2) 
        if dist > 0:
            speed = min(dist * 5, base_speed)
            self.x += (dx/dist) * speed * TICK_LEN
            self.y += (dy/dist) * speed * TICK_LEN
        self.x += self.boost_x * TICK_LEN
        self.y += self.boost_y * TICK_LEN
        self.boost_x *= FRICTION
        self.boost_y *= FRICTION
        
    def decay(self):
        if self.mass > BASE_MASS:
            self.mass -= self.mass * MASS_DECAY_RATE * TICK_LEN
            if self.mass < BASE_MASS: self.mass = BASE_MASS

class Player:
    # [Update] 新增 ip 參數
    def __init__(self, ws, pid, name, ip="Unknown", spectate=False):
        self.ws = ws
        self.id = pid
        self.name = name
        self.ip = ip  # 儲存 IP
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
        self.cells = [Cell(random.randint(100, MAP_WIDTH-100), random.randint(100, MAP_HEIGHT-100), BASE_MASS, self.color)]
        self.is_dead = False
        self.birth_time = time.time()
        self.max_mass_achieved = BASE_MASS

class Bot(Player):
    def __init__(self, pid, genes=None):
        bot_name = random.choice(BOT_NAMES)
        # Bot 的 IP 設為固定值
        super().__init__(None, pid, bot_name, ip="BOT-AI", spectate=False)
        self.color = "#%06x" % random.randint(0, 0xFFFFFF)
        self.genes = genes if genes else evo_manager.create_random_genes()

    def decide(self, world):
        if self.is_dead or not self.cells: return None
        current_mass = self.total_mass
        if current_mass > self.max_mass_achieved:
            self.max_mass_achieved = current_mass
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
        target_x += food_vec_x
        target_y += food_vec_y

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
                if dist > view_dist: continue
                if enemy_cell.mass > my_largest.mass * 1.15:
                    weight = w_flee / (dist + 1)
                    target_x += (dx / dist) * weight
                    target_y += (dy / dist) * weight
                elif enemy_cell.mass * split_aggr < my_largest.mass:
                    weight = w_hunt / (dist + 1)
                    target_x += (dx / dist) * weight
                    target_y += (dy / dist) * weight
                    if my_largest.mass > 50 and len(self.cells) < MAX_CELLS:
                        if dist < split_dist:
                            action_intent = 'split'
        w_virus = self.genes['w_virus']
        for v in world.viruses:
            dx = v.x - mx; dy = v.y - my
            dist = math.sqrt(dx**2 + dy**2)
            if dist > view_dist: continue
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
        self.max_food = 1200
        self.generate_food(800)
        self.generate_viruses(VIRUS_COUNT)

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
        for _ in range(amount):
            f = {'id': random.randint(0, 1000000000), 'x': random.randint(0,MAP_WIDTH), 'y': random.randint(0,MAP_HEIGHT), 'color': "#%06x"%random.randint(0,0xFFFFFF), 'mass': random.randint(5,10)}
            self.food.append(f)
            self.add_food_to_grid(f)
            
    def generate_viruses(self, amount):
        for _ in range(amount):
            self.viruses.append(Virus(random.randint(100, MAP_WIDTH-100), random.randint(100, MAP_HEIGHT-100)))

    def update(self):
        now = time.time()
        
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
                    if action == 'split':
                        self.event_queue.append({'type': GameEvent.SPLIT_CELL, 'player': p})
                    elif action == 'eject':
                        self.event_queue.append({'type': GameEvent.EJECT_MASS, 'player': p})
        # ------------------------

        active_players = [p for p in self.players.values() if not p.is_dead]

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

        for p1 in active_players:
            for i in range(len(p1.cells)):
                for j in range(i+1, len(p1.cells)):
                    c1, c2 = p1.cells[i], p1.cells[j]
                    dist = math.sqrt((c1.x-c2.x)**2 + (c1.y-c2.y)**2)
                    if dist < c1.radius + c2.radius:
                        if now > c1.recombine_time and now > c2.recombine_time and dist < (c1.radius+c2.radius)*0.65:
                            self.event_queue.append({'type': GameEvent.MERGE_CELLS, 'player': p1, 'idx1': i, 'idx2': j})
                        elif dist < c1.radius + c2.radius: 
                             pen = (c1.radius+c2.radius) - dist
                             if dist == 0: 
                                 rand_ang = random.uniform(0, math.pi*2)
                                 dx, dy = math.cos(rand_ang), math.sin(rand_ang)
                             else: 
                                 dx, dy = (c1.x-c2.x)/dist, (c1.y-c2.y)/dist
                             f = 0.5
                             c1.x += dx * pen * f
                             c1.y += dy * pen * f
                             c2.x -= dx * pen * f
                             c2.y -= dy * pen * f
            
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
        if len(self.viruses) < VIRUS_COUNT: self.generate_viruses(1)

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
                            c.set_recombine_cooldown()
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
                            pmass = c.mass / (pieces + 1)
                            c.mass = pmass; c.set_recombine_cooldown()
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
    """
    [Update] 新增查詢、增減質量、殺死玩家指令
    find [name]
    addmass [id] [amount]
    removemass [id] [amount]
    kill [id]
    killbotall
    """
    global pid_counter, MAP_WIDTH, MAP_HEIGHT
    
    command = cmd[0].lower()

    if command == "setsize" and len(cmd) == 3:
        try:
            w, h = int(cmd[1]), int(cmd[2])
            MAP_WIDTH, MAP_HEIGHT = w, h
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

    # --- [Update] 新增功能 ---
    
    elif command == "find":
        # 根據名字查詢 (部分匹配)
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
        # 增加指定玩家質量
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
        # 減少指定玩家質量
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
                        c.mass = max(10, c.mass - per_cell_loss) # 質量不低於 10
                    print(f"Removed {amount} mass from {p.name} (ID: {target_id}).")
                else:
                    print("Player not found or is dead.")
            except ValueError: print("Invalid ID or Amount.")

    elif command == "kill":
        # 殺死指定玩家
        if len(cmd) < 2:
            print("Usage: kill <player_id>")
        else:
            try:
                target_id = int(cmd[1])
                p = world.players.get(target_id)
                if p:
                    p.is_dead = True
                    p.cells = [] # 清空細胞
                    print(f"Killed player {p.name} (ID: {target_id}).")
                else:
                    print("Player not found.")
            except ValueError: print("Invalid ID.")

    elif command == "killbotall":
        # 殺死所有 BOT
        count = 0
        # 使用 list(values) 避免在迭代時刪除導致錯誤
        for p in list(world.players.values()):
            if isinstance(p, Bot):
                p.is_dead = True
                p.cells = []
                # 注意: 遊戲循環會自動處理移除，或在這裡直接從 world.players 移除也可以
                # 但為了讓死亡邏輯(基因紀錄)正常運作，我們只設為 is_dead
                count += 1
        print(f"Killed {count} bots.")

    else:
        print("Commands: setsize, clearfood, foodcfg, addbot, removebot, stats, find, addmass, removemass, kill, killbotall")

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
    
    # [Update] 獲取遠端 IP
    remote_ip = "Unknown"
    if ws.remote_address:
        remote_ip = f"{ws.remote_address[0]}:{ws.remote_address[1]}"

    try:
        async for msg in ws:
            data = json.loads(msg)
            if data['type'] == 'ping': await ws.send(json.dumps({'type':'pong', 'server_name':SERVER_NAME, 'players':len(world.players), 'max_players':MAX_PLAYERS}))
            elif data['type'] == 'join':
                # [Update] 傳入 IP
                player = Player(ws, pid, data.get('name', 'Guest')[:15], ip=remote_ip)
                world.players[pid] = player
                print(f"[Join] ID:{pid} Name:{player.name} IP:{remote_ip}") # 服務器日誌
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
    while True:
        t1 = time.time()
        world.update()
        
        active_players_snapshot = list(world.players.items())
        
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