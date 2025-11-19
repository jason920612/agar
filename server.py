import asyncio
import json
import math
import random
import time
import websockets
import sys
import aiohttp 
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# --- 伺服器設定 ---
SERVER_NAME = "Optimized Strat Server"
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
MASS_DECAY_RATE = 0.001 

# --- 物理與平衡參數 ---
BASE_MASS = 20
MAX_CELLS = 16
SPLIT_IMPULSE = 780    
EJECT_IMPULSE = 550
FRICTION = 0.90
VIRUS_START_MASS = 100
VIRUS_MAX_MASS = 180
VIRUS_COUNT = 50 # 稍微減少病毒數量以優化性能
VIRUS_SHOT_IMPULSE = 850

# --- 神經網路與進化參數 ---
INPUT_SIZE = 36   
HIDDEN_SIZE = 32  
OUTPUT_SIZE = 4   
MUTATION_RATE = 0.1 
MUTATION_STRENGTH = 0.4 

# 全域基因庫
BEST_BRAINS = {} 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mass_to_radius(mass):
    return 6 * math.sqrt(mass)

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

class SimpleBrain:
    def __init__(self, weights=None):
        if weights:
            self.w1, self.b1, self.w2, self.b2 = weights
        else:
            self.w1 = np.random.randn(INPUT_SIZE, HIDDEN_SIZE) * 0.5
            self.b1 = np.zeros(HIDDEN_SIZE)
            self.w2 = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE) * 0.5
            self.b2 = np.zeros(OUTPUT_SIZE)

    def forward(self, inputs):
        x = np.array(inputs)
        z1 = np.dot(x, self.w1) + self.b1
        a1 = np.maximum(0, z1) # ReLU
        z2 = np.dot(a1, self.w2) + self.b2
        move = np.tanh(z2[:2]) 
        actions = sigmoid(z2[2:])
        return np.concatenate((move, actions))

    def mutate(self):
        new_weights = [self.w1.copy(), self.b1.copy(), self.w2.copy(), self.b2.copy()]
        for param in new_weights:
            if random.random() < MUTATION_RATE:
                noise = np.random.randn(*param.shape) * MUTATION_STRENGTH
                param += noise
                if random.random() < 0.01:
                    idx = random.randint(0, param.size - 1)
                    param.flat[idx] = random.gauss(0, 1)
        return SimpleBrain(weights=new_weights)

class EjectedMass:
    def __init__(self, x, y, angle, color, parent_id, team_id=None):
        self.id = random.randint(100000, 999999)
        self.x = x
        self.y = y
        self.mass = 16 
        self.color = color
        self.radius = mass_to_radius(self.mass)
        self.vx = math.cos(angle) * EJECT_IMPULSE
        self.vy = math.sin(angle) * EJECT_IMPULSE
        self.parent_id = parent_id
        self.team_id = team_id
        self.birth_time = time.time()
        
    def move(self):
        self.x += self.vx * TICK_LEN
        self.y += self.vy * TICK_LEN
        self.vx *= FRICTION
        self.vy *= FRICTION
        self.x = clamp(self.x, 0, MAP_WIDTH)
        self.y = clamp(self.y, 0, MAP_HEIGHT)

class Virus:
    def __init__(self, x, y, mass=VIRUS_START_MASS, angle=0, velocity=0):
        self.id = random.randint(2000000, 9000000)
        self.x = x
        self.y = y
        self.mass = mass
        self.color = "#33ff33"
        self.vx = math.cos(angle) * velocity
        self.vy = math.sin(angle) * velocity
    
    @property
    def radius(self):
        return mass_to_radius(self.mass)

    def move(self):
        if self.vx != 0 or self.vy != 0:
            self.x += self.vx * TICK_LEN
            self.y += self.vy * TICK_LEN
            self.vx *= FRICTION
            self.vy *= FRICTION
            if abs(self.vx) < 1 and abs(self.vy) < 1:
                self.vx = 0
                self.vy = 0
            self.x = clamp(self.x, 0, MAP_WIDTH)
            self.y = clamp(self.y, 0, MAP_HEIGHT)

class Cell:
    def __init__(self, x, y, mass, color):
        self.id = random.randint(0, 10000000)
        self.x = x
        self.y = y
        self.mass = mass
        self.color = color
        self.boost_x = 0
        self.boost_y = 0
        self.set_recombine_cooldown()

    def set_recombine_cooldown(self):
        # ★★★ 修正：融合時間設定 ★★★
        # 基礎 30 秒 + (0.02 * 質量)
        # Mass 10 -> 30.2s
        # Mass 1000 -> 50.0s
        cooldown = 30 + (0.02 * self.mass)
        self.recombine_time = time.time() + cooldown

    @property
    def radius(self):
        return mass_to_radius(self.mass)

    def apply_force(self, force_x, force_y):
        self.boost_x += force_x
        self.boost_y += force_y

    def move(self, target_x, target_y):
        dx = target_x - self.x
        dy = target_y - self.y
        dist = math.sqrt(dx**2 + dy**2)
        base_speed = 300 * (self.mass ** -0.2)
        
        if dist > 0:
            dir_x = dx / dist
            dir_y = dy / dist
            speed = min(dist * 5, base_speed)
            self.x += dir_x * speed * TICK_LEN
            self.y += dir_y * speed * TICK_LEN

        self.x += self.boost_x * TICK_LEN
        self.y += self.boost_y * TICK_LEN
        self.boost_x *= FRICTION
        self.boost_y *= FRICTION
    
    def decay(self):
        if self.mass > BASE_MASS:
            loss = self.mass * MASS_DECAY_RATE * TICK_LEN
            self.mass -= loss
            if self.mass < BASE_MASS:
                self.mass = BASE_MASS

class Player:
    def __init__(self, websocket, pid, name, spectate=False):
        self.websocket = websocket
        self.id = pid
        self.name = name
        self.cells = []
        self.color = "#%06x" % random.randint(0, 0xFFFFFF)
        self.mouse_x = MAP_WIDTH / 2
        self.mouse_y = MAP_HEIGHT / 2
        self.is_dead = True
        self.is_spectator = spectate
        self.team_id = None 
        if not spectate:
            self.spawn()

    @property
    def total_mass(self):
        return sum([c.mass for c in self.cells])

    def spawn(self):
        if self.is_spectator: return
        self.cells = []
        self.is_dead = False
        start_x = random.randint(100, MAP_WIDTH-100)
        start_y = random.randint(100, MAP_HEIGHT-100)
        self.cells.append(Cell(start_x, start_y, BASE_MASS, self.color))

    def split(self):
        if self.is_dead: return
        new_cells = []
        current_count = len(self.cells)
        
        for cell in self.cells:
            if current_count >= MAX_CELLS: break
            if cell.mass < 35: continue

            split_mass = cell.mass / 2
            cell.mass = split_mass
            cell.set_recombine_cooldown()
            
            dx = self.mouse_x - cell.x
            dy = self.mouse_y - cell.y
            angle = math.atan2(dy, dx)

            new_cell = Cell(cell.x, cell.y, split_mass, cell.color)
            new_cell.apply_force(math.cos(angle) * SPLIT_IMPULSE, math.sin(angle) * SPLIT_IMPULSE)
            new_cells.append(new_cell)
            current_count += 1
        
        self.cells.extend(new_cells)

    def eject(self, world):
        if self.is_dead: return
        for cell in self.cells:
            if cell.mass < 35: continue
            cell.mass -= 16
            dx = self.mouse_x - cell.x
            dy = self.mouse_y - cell.y
            angle = math.atan2(dy, dx)
            
            start_x = cell.x + math.cos(angle) * cell.radius
            start_y = cell.y + math.sin(angle) * cell.radius
            ejected = EjectedMass(start_x, start_y, angle, cell.color, self.id, self.team_id)
            world.ejected_mass.append(ejected)

    def explode_on_virus(self, cell_index, virus_mass):
        target_cell = self.cells[cell_index]
        target_cell.mass += virus_mass
        if len(self.cells) >= MAX_CELLS: return
        if target_cell.mass < VIRUS_START_MASS * 1.2: return
        remaining_slots = MAX_CELLS - len(self.cells)
        if remaining_slots <= 0: return
        num_new_frags = min(remaining_slots, 7)
        total_pieces = num_new_frags + 1
        piece_mass = target_cell.mass / total_pieces
        target_cell.mass = piece_mass
        target_cell.set_recombine_cooldown()
        new_frags = []
        for i in range(num_new_frags):
            angle = (i / num_new_frags) * math.pi * 2
            frag = Cell(target_cell.x, target_cell.y, piece_mass, target_cell.color)
            speed = SPLIT_IMPULSE * 0.8
            frag.apply_force(math.cos(angle) * speed, math.sin(angle) * speed)
            new_frags.append(frag)
        self.cells.extend(new_frags)

class Bot(Player):
    def __init__(self, pid):
        self.team_id = random.randint(1, 4) 
        name = f"T{self.team_id}-Bot{random.randint(10,99)}"
        super().__init__(None, pid, name)
        
        team_colors = {1: "#FF3333", 2: "#33FF33", 3: "#3333FF", 4: "#FFFF33"}
        self.color = team_colors.get(self.team_id, "#FFFFFF")
        
        if self.team_id in BEST_BRAINS:
            parent_brain = BEST_BRAINS[self.team_id]['brain']
            self.brain = parent_brain.mutate()
            self.generation = BEST_BRAINS[self.team_id]['gen'] + 1
        else:
            self.brain = SimpleBrain() 
            self.generation = 1
            
        self.spawn_time = time.time()
        self.max_mass_achieved = BASE_MASS
        self.last_pos_check = time.time()
        self.last_pos_x = 0
        self.last_pos_y = 0
        self.stagnation_penalty = 0
        
        # ★★★ 優化：記錄上次思考時間 ★★★
        self.last_think_time = 0 

    def get_inputs(self, world):
        if not self.cells: return np.zeros(INPUT_SIZE)
        
        cx = sum(c.x for c in self.cells) / len(self.cells)
        cy = sum(c.y for c in self.cells) / len(self.cells)
        my_mass = self.total_mass
        if my_mass > self.max_mass_achieved: self.max_mass_achieved = my_mass

        # --- 1. 基礎資訊 ---
        norm_mass = min(my_mass / 10000, 1.0)
        dist_left = cx / MAP_WIDTH
        dist_right = (MAP_WIDTH - cx) / MAP_WIDTH
        dist_top = cy / MAP_HEIGHT
        dist_bottom = (MAP_HEIGHT - cy) / MAP_HEIGHT
        split_full = 1.0 if len(self.cells) >= MAX_CELLS else -1.0
        base_inputs = [norm_mass, dist_left, dist_right, dist_top, dist_bottom, split_full]

        # --- 2. 扇形視野 (優化版) ---
        sectors_food = np.zeros(8)
        sectors_threat = np.zeros(8)
        sectors_prey = np.zeros(8)
        
        vision_radius = 2000
        vision_radius_sq = vision_radius ** 2

        def get_sector_index(dx, dy):
            angle = math.atan2(dy, dx)
            if angle < 0: angle += 2 * math.pi
            idx = int((angle + math.pi/8) / (math.pi/4)) % 8
            return idx

        # 優化：不計算所有食物，只取部分或使用簡單距離檢查
        # 為了性能，這裡仍然遍歷，但在外部減少了食物總量
        for f in world.food:
            # 簡單矩形過濾 (比平方根快)
            if abs(f['x'] - cx) > vision_radius or abs(f['y'] - cy) > vision_radius:
                continue
            
            dx = f['x'] - cx
            dy = f['y'] - cy
            d_sq = dx*dx + dy*dy
            if d_sq < vision_radius_sq:
                idx = get_sector_index(dx, dy)
                sectors_food[idx] += 0.5

        nearest_teammate = None
        min_tm_dist = float('inf')
        nearest_enemy = None
        min_en_dist = float('inf')
        nearest_virus = None
        min_v_dist = float('inf')

        for p in world.players.values():
            if p.id == self.id or p.is_dead: continue
            
            # 計算玩家重心
            p_cx = sum(c.x for c in p.cells)/len(p.cells)
            p_cy = sum(c.y for c in p.cells)/len(p.cells)
            
            if abs(p_cx - cx) > vision_radius or abs(p_cy - cy) > vision_radius:
                continue

            dx = p_cx - cx
            dy = p_cy - cy
            dist_sq = dx*dx + dy*dy
            dist = math.sqrt(dist_sq)

            if p.team_id == self.team_id:
                if dist < min_tm_dist:
                    min_tm_dist = dist
                    nearest_teammate = p
            else:
                if dist < min_en_dist:
                    min_en_dist = dist
                    nearest_enemy = p
                
                idx = get_sector_index(dx, dy)
                if p.total_mass > my_mass * 1.2:
                    sectors_threat[idx] += p.total_mass / my_mass
                elif my_mass > p.total_mass * 1.2:
                    sectors_prey[idx] += p.total_mass / my_mass

        for v in world.viruses:
            if abs(v.x - cx) > vision_radius or abs(v.y - cy) > vision_radius: continue
            dx = v.x - cx
            dy = v.y - cy
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < min_v_dist:
                min_v_dist = dist
                nearest_virus = v

        # --- 3. 戰術特徵 ---
        combined_power = 0.0
        if nearest_teammate and nearest_enemy:
            if (my_mass + nearest_teammate.total_mass) > nearest_enemy.total_mass * 1.3:
                combined_power = 1.0
        
        alignment = 0.0
        if nearest_teammate and nearest_enemy:
            tm_cx = sum(c.x for c in nearest_teammate.cells)/len(nearest_teammate.cells)
            tm_cy = sum(c.y for c in nearest_teammate.cells)/len(nearest_teammate.cells)
            en_cx = sum(c.x for c in nearest_enemy.cells)/len(nearest_enemy.cells)
            en_cy = sum(c.y for c in nearest_enemy.cells)/len(nearest_enemy.cells)
            v1 = (tm_cx - cx, tm_cy - cy)
            v2 = (en_cx - tm_cx, en_cy - tm_cy)
            m1 = math.sqrt(v1[0]**2 + v1[1]**2)
            m2 = math.sqrt(v2[0]**2 + v2[1]**2)
            if m1 > 0 and m2 > 0:
                alignment = (v1[0]*v2[0] + v1[1]*v2[1]) / (m1 * m2)

        support_needed = 0.0
        if nearest_teammate and min_tm_dist < 600 and nearest_teammate.total_mass < my_mass * 0.4:
            support_needed = 1.0

        virus_shot_opp = 0.0
        if nearest_virus and nearest_enemy and min_v_dist < 700:
            en_cx = sum(c.x for c in nearest_enemy.cells)/len(nearest_enemy.cells)
            en_cy = sum(c.y for c in nearest_enemy.cells)/len(nearest_enemy.cells)
            v1 = (nearest_virus.x - cx, nearest_virus.y - cy)
            v2 = (en_cx - nearest_virus.x, en_cy - nearest_virus.y)
            m1 = math.sqrt(v1[0]**2 + v1[1]**2)
            m2 = math.sqrt(v2[0]**2 + v2[1]**2)
            if m1 > 0 and m2 > 0 and (v1[0]*v2[0] + v1[1]*v2[1]) / (m1 * m2) > 0.9:
                virus_shot_opp = 1.0

        tm_prox = 0.0
        if nearest_teammate:
            tm_prox = 1.0 - min(min_tm_dist / 1500, 1.0)

        feed_angle = 0.0
        if nearest_teammate and min_tm_dist < 400:
             feed_angle = 1.0

        tactical_inputs = [combined_power, alignment, support_needed, virus_shot_opp, tm_prox, feed_angle]
        
        return np.concatenate((base_inputs, np.tanh(sectors_food), np.tanh(sectors_threat), np.tanh(sectors_prey), tactical_inputs))

    def think(self, world):
        # ★★★ 性能優化核心：降低思考頻率 ★★★
        # 每個 Bot 每 0.15 秒才思考一次 (約每秒 6-7 次)，而不是每秒 20 次
        # 這大幅減少了 get_inputs 中的大量距離運算
        now = time.time()
        if now - self.last_think_time < 0.15: 
            return 
        self.last_think_time = now

        if self.is_dead:
            self.on_death(world)
            self.spawn()
            if self.team_id in BEST_BRAINS:
                self.brain = BEST_BRAINS[self.team_id]['brain'].mutate()
                self.generation = BEST_BRAINS[self.team_id]['gen'] + 1
            self.max_mass_achieved = BASE_MASS
            self.spawn_time = time.time()
            self.stagnation_penalty = 0
            return

        # 呆滯懲罰 (每 5 秒)
        if now - self.last_pos_check > 5.0:
            cx = sum(c.x for c in self.cells) / len(self.cells)
            cy = sum(c.y for c in self.cells) / len(self.cells)
            dist = math.sqrt((cx - self.last_pos_x)**2 + (cy - self.last_pos_y)**2)
            if dist < 200:
                self.stagnation_penalty += 1000
                self.split()
            self.last_pos_x = cx
            self.last_pos_y = cy
            self.last_pos_check = now

        inputs = self.get_inputs(world)
        outputs = self.brain.forward(inputs)
        
        move_x_raw, move_y_raw = outputs[0], outputs[1]
        do_split = outputs[2] > 0.7
        do_eject = outputs[3] > 0.7

        cx = sum(c.x for c in self.cells) / len(self.cells)
        cy = sum(c.y for c in self.cells) / len(self.cells)
        
        # 牆壁斥力
        wall_force_x = 0
        wall_force_y = 0
        margin = 300
        if cx < margin: wall_force_x = 1.0
        if cx > MAP_WIDTH - margin: wall_force_x = -1.0
        if cy < margin: wall_force_y = 1.0
        if cy > MAP_HEIGHT - margin: wall_force_y = -1.0
        
        final_dir_x = move_x_raw + wall_force_x * 2.5
        final_dir_y = move_y_raw + wall_force_y * 2.5
        
        self.mouse_x = cx + final_dir_x * 1000
        self.mouse_y = cy + final_dir_y * 1000
        
        if do_split and self.total_mass > 36 and len(self.cells) < MAX_CELLS:
            self.split()
        
        if do_eject and self.total_mass > 36:
            self.eject(world)

    def on_death(self, world):
        lifespan = time.time() - self.spawn_time
        team_total_mass = 0
        for p in world.players.values():
            if p.team_id == self.team_id and not p.is_dead:
                team_total_mass += p.total_mass
        
        score = (self.max_mass_achieved * 1.0) + (team_total_mass * 0.8) + (lifespan * 0.5) - self.stagnation_penalty
        
        current_best = BEST_BRAINS.get(self.team_id)
        if current_best is None or score > current_best['score']:
            BEST_BRAINS[self.team_id] = {
                'brain': self.brain, 
                'score': score,
                'gen': self.generation
            }

class GameWorld:
    def __init__(self):
        self.players = {}
        self.food = []
        self.viruses = []
        self.ejected_mass = []
        # ★★★ 優化：減少最大食物數量 (減少迴圈運算)，增加單體質量 ★★★
        self.max_food = 800 # 原本 1500 -> 800，大幅減少延遲
        self.food_spawn_rate = 10
        self.events = []
        self.generate_food(500)
        self.generate_viruses(VIRUS_COUNT)

    def generate_food(self, amount):
        for _ in range(amount):
            self.food.append({
                'id': random.randint(0, 1000000),
                'x': random.randint(0, MAP_WIDTH),
                'y': random.randint(0, MAP_HEIGHT),
                'color': "#%06x" % random.randint(0, 0xFFFFFF),
                # 增加質量補償數量減少
                'mass': random.randint(5, 10) 
            })
            
    def generate_viruses(self, amount):
        for _ in range(amount):
            self.viruses.append(Virus(random.randint(100, MAP_WIDTH-100), random.randint(100, MAP_HEIGHT-100)))

    def resolve_player_collisions_and_merge(self, player):
        now = time.time()
        to_remove_indices = set()
        
        for i in range(len(player.cells)):
            if i in to_remove_indices: continue
            for j in range(i + 1, len(player.cells)):
                if j in to_remove_indices: continue
                c1 = player.cells[i]
                c2 = player.cells[j]
                dx = c1.x - c2.x
                dy = c1.y - c2.y
                dist = math.sqrt(dx*dx + dy*dy)
                radius_sum = c1.radius + c2.radius
                can_merge = (now > c1.recombine_time) and (now > c2.recombine_time)
                
                if can_merge:
                    if dist < radius_sum * 0.65:
                        c1.mass += c2.mass
                        c1.x = (c1.x * c1.mass + c2.x * c2.mass) / (c1.mass + c2.mass)
                        c1.y = (c1.y * c1.mass + c2.y * c2.mass) / (c1.mass + c2.mass)
                        to_remove_indices.add(j)
                    else:
                        if dist > 0:
                            force = 30 * TICK_LEN
                            nx, ny = dx/dist, dy/dist
                            c1.x -= nx * force
                            c1.y -= ny * force
                            c2.x += nx * force
                            c2.y += ny * force
                else:
                    if dist < radius_sum:
                        if dist == 0: dist, dx = 1, 1
                        penetration = radius_sum - dist
                        nx, ny = dx/dist, dy/dist
                        force = penetration * 1.0 
                        c1.x += nx * force
                        c1.y += ny * force
                        c2.x -= nx * force
                        c2.y -= ny * force

        if to_remove_indices:
            new_cells = []
            for idx, cell in enumerate(player.cells):
                if idx not in to_remove_indices:
                    new_cells.append(cell)
            player.cells = new_cells

    def update(self):
        now = time.time()
        for p in self.players.values():
            if isinstance(p, Bot): p.think(self)

        for e in self.ejected_mass: e.move()
        for v in self.viruses: v.move()
        
        active_players = [p for p in self.players.values() if not p.is_dead]
        
        for p in active_players:
            for cell in p.cells:
                cell.move(p.mouse_x, p.mouse_y)
                cell.decay()
            self.resolve_player_collisions_and_merge(p)
            for cell in p.cells:
                cell.x = clamp(cell.x, 0, MAP_WIDTH)
                cell.y = clamp(cell.y, 0, MAP_HEIGHT)

        ejects_to_remove = set()
        viruses_to_add = []
        
        for v_idx, v in enumerate(self.viruses):
            for e_idx, e in enumerate(self.ejected_mass):
                if e_idx in ejects_to_remove: continue
                dx, dy = e.x - v.x, e.y - v.y
                if dx*dx + dy*dy < (v.radius + e.radius)**2:
                    v.mass += e.mass
                    ejects_to_remove.add(e_idx)
                    if v.mass >= VIRUS_MAX_MASS:
                        v.mass = VIRUS_START_MASS
                        shot_angle = math.atan2(e.vy, e.vx)
                        start_dist = v.radius * 2
                        viruses_to_add.append(Virus(
                            v.x + math.cos(shot_angle) * start_dist,
                            v.y + math.sin(shot_angle) * start_dist,
                            mass=VIRUS_START_MASS, angle=shot_angle, velocity=VIRUS_SHOT_IMPULSE
                        ))

        self.ejected_mass = [e for i, e in enumerate(self.ejected_mass) if i not in ejects_to_remove]
        self.viruses.extend(viruses_to_add)

        for p in active_players:
            virus_hit_info = [] 
            for i, cell in enumerate(p.cells):
                for idx in range(len(self.food) - 1, -1, -1):
                    f = self.food[idx]
                    if (cell.x-f['x'])**2 + (cell.y-f['y'])**2 < cell.radius**2:
                        cell.mass += f['mass']
                        del self.food[idx]
                
                for idx in range(len(self.ejected_mass) - 1, -1, -1):
                    e = self.ejected_mass[idx]
                    if e.parent_id == p.id and (now - e.birth_time < 0.2): continue
                    if (cell.x-e.x)**2 + (cell.y-e.y)**2 < cell.radius**2:
                        cell.mass += e.mass
                        del self.ejected_mass[idx]
                
                for v_idx in range(len(self.viruses) - 1, -1, -1):
                    v = self.viruses[v_idx]
                    if cell.mass > v.mass * 1.1:
                         if (cell.x - v.x)**2 + (cell.y - v.y)**2 < cell.radius**2:
                            virus_hit_info.append((i, v.mass))
                            del self.viruses[v_idx]
                            break
            
            virus_hit_info.sort(key=lambda x: x[0], reverse=True)
            for idx, v_mass in virus_hit_info:
                p.explode_on_virus(idx, v_mass)

        for p1 in active_players:
            for cell1 in p1.cells:
                for p2 in active_players:
                    if p1.id == p2.id: continue
                    
                    is_teammate = (p1.team_id == p2.team_id)
                    
                    for cell2 in p2.cells:
                        dist = math.sqrt((cell1.x - cell2.x)**2 + (cell1.y - cell2.y)**2)
                        if dist < cell1.radius:
                            can_eat = False
                            if is_teammate:
                                if cell1.mass > cell2.mass * 1.30: 
                                    can_eat = True
                            else:
                                if cell1.mass > cell2.mass * 1.25:
                                    can_eat = True
                            
                            if can_eat:
                                cell1.mass += cell2.mass
                                cell2.mass = 0
                    
                    p2.cells = [c for c in p2.cells if c.mass > 0]
                    if len(p2.cells) == 0: p2.is_dead = True

        if len(self.food) < self.max_food: self.generate_food(self.food_spawn_rate)
        if len(self.viruses) < VIRUS_COUNT: self.generate_viruses(1)

    def get_state(self):
        lb_data = [{'id': p.id, 'name': p.name, 'mass': int(p.total_mass)} 
                   for p in sorted([p for p in self.players.values() if not p.is_dead and not p.is_spectator], 
                   key=lambda p: p.total_mass, reverse=True)[:10]]
                   
        return {
            'players': [{'id': p.id, 'name': p.name, 'dead': p.is_dead, 
                         'cells': [{'x': int(c.x), 'y': int(c.y), 'm': int(c.mass), 'c': c.color} for c in p.cells]} 
                        for p in self.players.values() if not p.is_spectator],
            'food': self.food,
            'viruses': [{'x': int(v.x), 'y': int(v.y), 'm': int(v.mass)} for v in self.viruses],
            'ejected': [{'x': int(e.x), 'y': int(e.y), 'c': e.color} for e in self.ejected_mass],
            'leaderboard': lb_data
        }

world = GameWorld()
pid_counter = 0

async def register_to_master():
    try:
        async with aiohttp.ClientSession() as session:
            data = {'url': MY_URL, 'name': SERVER_NAME, 'max_players': MAX_PLAYERS}
            async with session.post(f"{MASTER_URL}/register", json=data) as resp:
                print("Registered to Master")
    except Exception as e:
        print(f"Master Error: {e}")

async def deregister_from_master():
    try:
        async with aiohttp.ClientSession() as session:
            await session.post(f"{MASTER_URL}/deregister", json={'url': MY_URL})
    except: pass

def manage_game_commands(cmd):
    global pid_counter, MAP_WIDTH, MAP_HEIGHT, BEST_BRAINS
    
    if cmd[0] == "addbot" and len(cmd) > 1:
        try:
            n = int(cmd[1])
            print(f"Adding {n} Smart Bots...")
            for _ in range(n):
                pid = pid_counter
                pid_counter += 1
                world.players[pid] = Bot(pid)
        except ValueError: print("Invalid number")
        
    elif cmd[0] == "resetbrains":
        BEST_BRAINS = {}
        print("Brains reset.")

    elif cmd[0] == "removebot" and len(cmd) > 1:
        try:
            n = int(cmd[1])
            bots = [pid for pid, p in world.players.items() if isinstance(p, Bot)]
            for pid in bots[:n]: del world.players[pid]
        except ValueError: print("Error")

    elif cmd[0] == "setsize" and len(cmd) == 3:
        try:
            w, h = int(cmd[1]), int(cmd[2])
            MAP_WIDTH, MAP_HEIGHT = w, h
            world.events.append({'type': 'map_update', 'map': {'w': w, 'h': h}})
            print(f"Map size updated to {w}x{h}")
        except ValueError: print("Usage: setsize <width> <height>")

    elif cmd[0] == "clearfood":
        world.food = []
        print("All food cleared.")

    elif cmd[0] == "foodcfg" and len(cmd) == 3:
        try:
            max_f, rate = int(cmd[1]), int(cmd[2])
            world.max_food = max_f
            world.food_spawn_rate = rate
            print(f"Food Config Updated: Max={max_f}, Rate={rate}")
        except ValueError: print("Usage: foodcfg <max_amount> <spawn_rate>")

    else:
        print("Commands: addbot, removebot, resetbrains, setsize, clearfood, foodcfg")

async def input_loop():
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(1, "InputThread") as executor:
        while True:
            cmd = await loop.run_in_executor(executor, sys.stdin.readline)
            cmd = cmd.strip().split()
            if not cmd: continue
            manage_game_commands(cmd)

async def handler(websocket):
    global pid_counter
    pid = pid_counter
    pid_counter += 1
    current_player = None
    try:
        async for message in websocket:
            data = json.loads(message)
            if data['type'] == 'ping':
                await websocket.send(json.dumps({'type':'pong', 'server_name': SERVER_NAME, 'players': len(world.players), 'max_players': MAX_PLAYERS}))
            elif data['type'] == 'join':
                current_player = Player(websocket, pid, data.get('name', 'Guest')[:15])
                world.players[pid] = current_player
                await websocket.send(json.dumps({'type': 'init', 'id': pid, 'map': {'w': MAP_WIDTH, 'h': MAP_HEIGHT}}))
            elif data['type'] == 'spectate':
                current_player = Player(websocket, pid, "Spectator", spectate=True)
                world.players[pid] = current_player
                await websocket.send(json.dumps({'type': 'init', 'id': pid, 'map': {'w': MAP_WIDTH, 'h': MAP_HEIGHT}}))
            elif current_player and not current_player.is_spectator:
                if data['type'] == 'input':
                    current_player.mouse_x = data['x']
                    current_player.mouse_y = data['y']
                elif data['type'] == 'split': current_player.split()
                elif data['type'] == 'eject': current_player.eject(world)
                elif data['type'] == 'respawn' and current_player.is_dead: current_player.spawn()
    except: pass
    finally:
        if pid in world.players and not isinstance(world.players[pid], Bot):
            del world.players[pid]

async def game_loop():
    while True:
        start = time.time()
        world.update()
        msg_state = json.dumps({'type': 'update', 'data': world.get_state()})
        msg_events = []
        while world.events:
            event = world.events.pop(0)
            msg_events.append(json.dumps(event))

        to_del = []
        for pid, p in world.players.items():
            if isinstance(p, Bot): continue
            try: 
                await p.websocket.send(msg_state)
                for event_msg in msg_events:
                    await p.websocket.send(event_msg)
            except: to_del.append(pid)
        for pid in to_del: del world.players[pid]
        await asyncio.sleep(max(0, TICK_LEN - (time.time() - start)))

async def main():
    print(f"Optimized Neural Agar.io Server - {SERVER_NAME}")
    await register_to_master()
    try:
        await asyncio.gather(game_loop(), websockets.serve(handler, SERVER_HOST, SERVER_PORT), input_loop())
    finally:
        await deregister_from_master()

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: asyncio.run(deregister_from_master())