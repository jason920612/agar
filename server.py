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
SERVER_NAME = "Pro-Team Strat Neural Server"
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
MASS_DECAY_RATE = 0.001 # 極低的衰減，鼓勵囤積質量

# --- 物理與平衡參數 ---
BASE_MASS = 20
MAX_CELLS = 16
SPLIT_IMPULSE = 780    # 增加分裂推力，讓 Tricksplit 更猛
EJECT_IMPULSE = 550
FRICTION = 0.90
VIRUS_START_MASS = 100
VIRUS_MAX_MASS = 180
VIRUS_COUNT = 60
VIRUS_SHOT_IMPULSE = 850

# --- 神經網路與進化參數 ---
# 輸入層擴增至 36：
# 6(基礎) + 24(扇形視野) + 6(高階戰術特徵)
INPUT_SIZE = 36   
HIDDEN_SIZE = 32  # 增加大腦容量以處理複雜戰術
OUTPUT_SIZE = 4   # MoveX, MoveY, Split, Eject
MUTATION_RATE = 0.1 
MUTATION_STRENGTH = 0.4 # 強突變，嘗試激進策略

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
        # Layer 1
        z1 = np.dot(x, self.w1) + self.b1
        a1 = np.maximum(0, z1) # ReLU
        
        # Layer 2
        z2 = np.dot(a1, self.w2) + self.b2
        
        # Outputs
        move = np.tanh(z2[:2]) 
        actions = sigmoid(z2[2:])
        
        return np.concatenate((move, actions))

    def mutate(self):
        new_weights = [self.w1.copy(), self.b1.copy(), self.w2.copy(), self.b2.copy()]
        for param in new_weights:
            if random.random() < MUTATION_RATE:
                noise = np.random.randn(*param.shape) * MUTATION_STRENGTH
                param += noise
                # 隨機重置神經元連接，尋找新的戰術路徑
                if random.random() < 0.01:
                    idx = random.randint(0, param.size - 1)
                    param.flat[idx] = random.gauss(0, 1)
        return SimpleBrain(weights=new_weights)

class EjectedMass:
    def __init__(self, x, y, angle, color, parent_id, team_id=None):
        self.id = random.randint(100000, 999999)
        self.x = x
        self.y = y
        self.mass = 16 # 稍微增加 W 的質量
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
        # 加快合球速度，方便快速傳遞質量
        cooldown = 5 + (0.01 * self.mass) # 原本是 30 + 0.02
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

    def get_inputs(self, world):
        if not self.cells: return np.zeros(INPUT_SIZE)
        
        cx = sum(c.x for c in self.cells) / len(self.cells)
        cy = sum(c.y for c in self.cells) / len(self.cells)
        my_mass = self.total_mass
        if my_mass > self.max_mass_achieved: self.max_mass_achieved = my_mass

        # --- 1. 基礎資訊 (6) ---
        norm_mass = min(my_mass / 10000, 1.0)
        dist_left = cx / MAP_WIDTH
        dist_right = (MAP_WIDTH - cx) / MAP_WIDTH
        dist_top = cy / MAP_HEIGHT
        dist_bottom = (MAP_HEIGHT - cy) / MAP_HEIGHT
        split_full = 1.0 if len(self.cells) >= MAX_CELLS else -1.0
        base_inputs = [norm_mass, dist_left, dist_right, dist_top, dist_bottom, split_full]

        # --- 2. 扇形視野 (24) ---
        sectors_food = np.zeros(8)
        sectors_threat = np.zeros(8) # 比我大的敵人
        sectors_prey = np.zeros(8)   # 比我小的敵人 (可吃)
        
        vision_radius = 2000

        def get_sector_index(dx, dy):
            angle = math.atan2(dy, dx)
            if angle < 0: angle += 2 * math.pi
            idx = int((angle + math.pi/8) / (math.pi/4)) % 8
            return idx

        for f in world.food:
            dx = f['x'] - cx
            dy = f['y'] - cy
            d_sq = dx*dx + dy*dy
            if d_sq < vision_radius**2:
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
            
            p_cx = sum(c.x for c in p.cells)/len(p.cells)
            p_cy = sum(c.y for c in p.cells)/len(p.cells)
            dx = p_cx - cx
            dy = p_cy - cy
            dist = math.sqrt(dx**2 + dy**2)

            # 紀錄最近的隊友/敵人供後續特徵使用
            if p.team_id == self.team_id:
                if dist < min_tm_dist:
                    min_tm_dist = dist
                    nearest_teammate = p
                # 隊友不計入視野威脅/獵物，而是單獨處理
            else:
                if dist < min_en_dist:
                    min_en_dist = dist
                    nearest_enemy = p
                
                if dist < vision_radius:
                    idx = get_sector_index(dx, dy)
                    if p.total_mass > my_mass * 1.2:
                        sectors_threat[idx] += p.total_mass / my_mass
                    elif my_mass > p.total_mass * 1.2:
                        sectors_prey[idx] += p.total_mass / my_mass

        for v in world.viruses:
            dx = v.x - cx
            dy = v.y - cy
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < min_v_dist:
                min_v_dist = dist
                nearest_virus = v
            # 病毒處理略... (同上)

        # --- 3. 高階戰術特徵 (6) ---
        
        # 特徵 A: 聯合力量 (Combined Power)
        # 如果 (我+隊友) > 敵人 * 1.3，這是一個進攻訊號
        combined_power = 0.0
        if nearest_teammate and nearest_enemy:
            tm_mass = nearest_teammate.total_mass
            en_mass = nearest_enemy.total_mass
            if (my_mass + tm_mass) > en_mass * 1.3:
                combined_power = 1.0
        
        # 特徵 B: 隊友對齊度 (Teammate Alignment)
        # 向量點積：檢查 "我->隊友" 的方向是否與 "隊友->敵人" 的方向一致
        # 如果一致，代表我們排成一列，適合 Tricksplit
        alignment = 0.0
        if nearest_teammate and nearest_enemy:
            tm_cx = sum(c.x for c in nearest_teammate.cells)/len(nearest_teammate.cells)
            tm_cy = sum(c.y for c in nearest_teammate.cells)/len(nearest_teammate.cells)
            en_cx = sum(c.x for c in nearest_enemy.cells)/len(nearest_enemy.cells)
            en_cy = sum(c.y for c in nearest_enemy.cells)/len(nearest_enemy.cells)
            
            v_me_tm = (tm_cx - cx, tm_cy - cy) # 我到隊友
            v_tm_en = (en_cx - tm_cx, en_cy - tm_cy) # 隊友到敵人
            
            # 歸一化
            mag1 = math.sqrt(v_me_tm[0]**2 + v_me_tm[1]**2)
            mag2 = math.sqrt(v_tm_en[0]**2 + v_tm_en[1]**2)
            if mag1 > 0 and mag2 > 0:
                dot_product = (v_me_tm[0]*v_tm_en[0] + v_me_tm[1]*v_tm_en[1]) / (mag1 * mag2)
                alignment = dot_product # 1.0 代表完美直線

        # 特徵 C: 隊友需要餵食 (Support Needed)
        # 如果隊友比我小很多，且就在附近
        support_needed = 0.0
        if nearest_teammate and min_tm_dist < 600:
            if nearest_teammate.total_mass < my_mass * 0.4:
                support_needed = 1.0

        # 特徵 D: 病毒攻擊機會 (Virus Shot)
        # 我 + 病毒 + 敵人 連成一線
        virus_shot_opp = 0.0
        if nearest_virus and nearest_enemy and min_v_dist < 700:
            v_x, v_y = nearest_virus.x, nearest_virus.y
            en_x = sum(c.x for c in nearest_enemy.cells)/len(nearest_enemy.cells)
            en_y = sum(c.y for c in nearest_enemy.cells)/len(nearest_enemy.cells)
            
            v1 = (v_x - cx, v_y - cy)
            v2 = (en_x - v_x, en_y - v_y)
            mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
            if mag1 > 0 and mag2 > 0:
                if (v1[0]*v2[0] + v1[1]*v2[1]) / (mag1 * mag2) > 0.9: # 角度很正
                    virus_shot_opp = 1.0

        # 特徵 E: 隊友距離 (Teammate Proximity)
        tm_prox = 0.0
        if nearest_teammate:
            tm_prox = 1.0 - min(min_tm_dist / 1500, 1.0)

        # 特徵 F: 是否在隊友的「嘴邊」 (Feeding Angle)
        # 用於判斷是否應該分裂撞向隊友
        feed_angle = 0.0
        if nearest_teammate and min_tm_dist < 400:
             feed_angle = 1.0

        tactical_inputs = [combined_power, alignment, support_needed, virus_shot_opp, tm_prox, feed_angle]
        
        final_input = np.concatenate((base_inputs, np.tanh(sectors_food), np.tanh(sectors_threat), np.tanh(sectors_prey), tactical_inputs))
        return final_input

    def think(self, world):
        if self.is_dead:
            self.on_death(world) # 傳入 world 以計算隊伍總分
            self.spawn()
            if self.team_id in BEST_BRAINS:
                self.brain = BEST_BRAINS[self.team_id]['brain'].mutate()
                self.generation = BEST_BRAINS[self.team_id]['gen'] + 1
            self.max_mass_achieved = BASE_MASS
            self.spawn_time = time.time()
            self.stagnation_penalty = 0
            return

        # 呆滯檢查
        now = time.time()
        if now - self.last_pos_check > 5.0:
            cx = sum(c.x for c in self.cells) / len(self.cells)
            cy = sum(c.y for c in self.cells) / len(self.cells)
            dist = math.sqrt((cx - self.last_pos_x)**2 + (cy - self.last_pos_y)**2)
            if dist < 200:
                self.stagnation_penalty += 1000 # 重罰
                self.split() # 強制動作
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
        
        # 行動執行
        if do_split and self.total_mass > 36 and len(self.cells) < MAX_CELLS:
            # 降低隨機性，讓神經網路全權決定
            # 如果輸入顯示 "Combined Power" 高，這裡應該會觸發
            self.split()
        
        if do_eject and self.total_mass > 36:
            self.eject(world)

    def on_death(self, world):
        lifespan = time.time() - self.spawn_time
        
        # ★★★ 核心修改：計算隊伍總質量 ★★★
        team_total_mass = 0
        for p in world.players.values():
            if p.team_id == self.team_id and not p.is_dead:
                team_total_mass += p.total_mass
        
        # 分數 = 個人成就 + (隊伍成就 * 0.8)
        # 這意味著：如果我死了，但我隊友現在有 10000 分，我也會得到很高的適應度分數
        # 這會鼓勵 Bot 做出犧牲行為 (Tricksplit 餵給隊友)
        score = (self.max_mass_achieved * 1.0) + (team_total_mass * 0.8) + (lifespan * 0.5) - self.stagnation_penalty
        
        current_best = BEST_BRAINS.get(self.team_id)
        if current_best is None or score > current_best['score']:
            BEST_BRAINS[self.team_id] = {
                'brain': self.brain, 
                'score': score,
                'gen': self.generation
            }
            # print(f"Team {self.team_id} Gen {self.generation} Score {int(score)} (TeamMass contribution: {int(team_total_mass)})")

class GameWorld:
    def __init__(self):
        self.players = {}
        self.food = []
        self.viruses = []
        self.ejected_mass = []
        self.max_food = 1500
        self.food_spawn_rate = 20
        self.events = []
        self.generate_food(800)
        self.generate_viruses(VIRUS_COUNT)

    def generate_food(self, amount):
        for _ in range(amount):
            self.food.append({
                'id': random.randint(0, 1000000),
                'x': random.randint(0, MAP_WIDTH),
                'y': random.randint(0, MAP_HEIGHT),
                'color': "#%06x" % random.randint(0, 0xFFFFFF),
                'mass': random.randint(3, 8)
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
                        if dist > 0: # 內部吸力
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
                        # 減少內部排斥力，允許細胞貼得更近，方便合球
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
                # 食物
                for idx in range(len(self.food) - 1, -1, -1):
                    f = self.food[idx]
                    if (cell.x-f['x'])**2 + (cell.y-f['y'])**2 < cell.radius**2:
                        cell.mass += f['mass']
                        del self.food[idx]
                
                # 吐出質量
                for idx in range(len(self.ejected_mass) - 1, -1, -1):
                    e = self.ejected_mass[idx]
                    if e.parent_id == p.id and (now - e.birth_time < 0.2): continue
                    if (cell.x-e.x)**2 + (cell.y-e.y)**2 < cell.radius**2:
                        cell.mass += e.mass
                        del self.ejected_mass[idx]
                
                # 病毒
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

        # ★★★ 吞噬邏輯 (允許隊友互吃) ★★★
        for p1 in active_players:
            for cell1 in p1.cells:
                for p2 in active_players:
                    if p1.id == p2.id: continue
                    
                    # 判斷是否為隊友
                    is_teammate = (p1.team_id == p2.team_id)
                    
                    for cell2 in p2.cells:
                        dist = math.sqrt((cell1.x - cell2.x)**2 + (cell1.y - cell2.y)**2)
                        if dist < cell1.radius:
                            # 吃掉的條件
                            can_eat = False
                            
                            if is_teammate:
                                # 隊友互吃條件：
                                # 1. 差距夠大 (大吃小加速)
                                # 2. 或者被吃者剛出生不久 (剛分裂出來的小球)
                                if cell1.mass > cell2.mass * 1.30: # 隊友需要 30% 差距
                                    can_eat = True
                            else:
                                # 敵人互吃：25% 差距
                                if cell1.mass > cell2.mass * 1.25:
                                    can_eat = True
                            
                            if can_eat:
                                cell1.mass += cell2.mass
                                cell2.mass = 0 # 標記為死亡
                    
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
    print(f"Pro-Team Neural Agar.io Server - {SERVER_NAME}")
    await register_to_master()
    try:
        await asyncio.gather(game_loop(), websockets.serve(handler, SERVER_HOST, SERVER_PORT), input_loop())
    finally:
        await deregister_from_master()

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: asyncio.run(deregister_from_master())