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
SERVER_NAME = "Stable Event-Driven Server"
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
VIRUS_COUNT = 50 
VIRUS_SHOT_IMPULSE = 850

# --- 視野優化參數 ---
GRID_SIZE = 300 
VIEW_W = 2500 
VIEW_H = 1800

# --- 神經網路參數 ---
INPUT_SIZE = 36   
HIDDEN_SIZE = 32  
OUTPUT_SIZE = 4   
MUTATION_RATE = 0.1 
MUTATION_STRENGTH = 0.4 
BEST_BRAINS = {} 

# --- 輔助函數 ---
def sigmoid(x): return 1 / (1 + np.exp(-x))
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

# --- 類別定義 ---
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
        a1 = np.maximum(0, z1) 
        z2 = np.dot(a1, self.w2) + self.b2
        return np.concatenate((np.tanh(z2[:2]), sigmoid(z2[2:])))
    def mutate(self):
        new_weights = [self.w1.copy(), self.b1.copy(), self.w2.copy(), self.b2.copy()]
        for param in new_weights:
            if random.random() < MUTATION_RATE:
                param += np.random.randn(*param.shape) * MUTATION_STRENGTH
                if random.random() < 0.01:
                    idx = random.randint(0, param.size - 1)
                    param.flat[idx] = random.gauss(0, 1)
        return SimpleBrain(weights=new_weights)

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
        # ★★★ 防呆：如果目標座標無效，就不移動 ★★★
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
    def __init__(self, ws, pid, name, spectate=False):
        self.ws = ws
        self.id = pid
        self.name = name
        self.cells = []
        self.color = "#%06x" % random.randint(0, 0xFFFFFF)
        self.mouse_x, self.mouse_y = MAP_WIDTH/2, MAP_HEIGHT/2
        self.is_dead = True
        self.is_spectator = spectate
        self.team_id = None
        if not spectate: self.spawn()
    @property
    def total_mass(self): return sum([c.mass for c in self.cells])
    @property
    def center(self):
        if not self.cells: return (MAP_WIDTH/2, MAP_HEIGHT/2)
        return (sum(c.x for c in self.cells)/len(self.cells), sum(c.y for c in self.cells)/len(self.cells))
    def spawn(self):
        if self.is_spectator: return
        self.cells = [Cell(random.randint(100, MAP_WIDTH-100), random.randint(100, MAP_HEIGHT-100), BASE_MASS, self.color)]
        self.is_dead = False

class Bot(Player):
    def __init__(self, pid):
        self.team_id = random.randint(1, 4)
        super().__init__(None, pid, f"T{self.team_id}-Bot{random.randint(10,99)}")
        self.color = {1:"#FF3333", 2:"#33FF33", 3:"#3333FF", 4:"#FFFF33"}.get(self.team_id, "#FFF")
        self.brain = BEST_BRAINS.get(self.team_id, {}).get('brain', SimpleBrain()).mutate()
        self.generation = BEST_BRAINS.get(self.team_id, {}).get('gen', 0) + 1
        self.spawn_time = time.time()
        self.max_mass_achieved = BASE_MASS
        self.last_think_time = 0
        self.last_pos = (0,0); self.last_pos_check = time.time(); self.stag_penalty = 0

    def think(self, world):
        now = time.time()
        if now - self.last_think_time < 0.15: return
        self.last_think_time = now
        
        if self.is_dead:
            self.on_death(world)
            self.spawn()
            self.brain = BEST_BRAINS.get(self.team_id, {}).get('brain', self.brain).mutate()
            self.max_mass_achieved = BASE_MASS; self.spawn_time = now; self.stag_penalty = 0
            return

        cx, cy = self.center
        if now - self.last_pos_check > 5.0:
            if math.sqrt((cx-self.last_pos[0])**2 + (cy-self.last_pos[1])**2) < 200:
                self.stag_penalty += 1000
                world.event_queue.append({'type': GameEvent.SPLIT_CELL, 'player': self})
            self.last_pos = (cx, cy); self.last_pos_check = now

        inputs = np.zeros(INPUT_SIZE) 
        out = self.brain.forward(inputs)
        self.mouse_x = cx + (out[0] + (1 if cx < 300 else -1 if cx > MAP_WIDTH-300 else 0)*2) * 1000
        self.mouse_y = cy + (out[1] + (1 if cy < 300 else -1 if cy > MAP_HEIGHT-300 else 0)*2) * 1000
        
        if out[2] > 0.7 and self.total_mass > 36: world.event_queue.append({'type': GameEvent.SPLIT_CELL, 'player': self})
        if out[3] > 0.7 and self.total_mass > 36: world.event_queue.append({'type': GameEvent.EJECT_MASS, 'player': self})

    def on_death(self, world):
        score = self.max_mass_achieved + sum(p.total_mass for p in world.players.values() if p.team_id==self.team_id)*0.8 + (time.time()-self.spawn_time)*0.5 - self.stag_penalty
        best = BEST_BRAINS.get(self.team_id)
        if not best or score > best['score']: BEST_BRAINS[self.team_id] = {'brain': self.brain, 'score': score, 'gen': self.generation}

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
        
        for p in self.players.values():
            if isinstance(p, Bot): p.think(self)
            
        for e in self.ejected_mass: e.move()
        for v in self.viruses: v.move()
        
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
            to_merge = []
            for i in range(len(p1.cells)):
                for j in range(i+1, len(p1.cells)):
                    c1, c2 = p1.cells[i], p1.cells[j]
                    dist = math.sqrt((c1.x-c2.x)**2 + (c1.y-c2.y)**2)
                    if dist < c1.radius + c2.radius:
                        if now > c1.recombine_time and now > c2.recombine_time and dist < (c1.radius+c2.radius)*0.65:
                            self.event_queue.append({'type': GameEvent.MERGE_CELLS, 'player': p1, 'idx1': i, 'idx2': j})
                        elif dist < c1.radius + c2.radius: 
                             pen = (c1.radius+c2.radius) - dist
                             if dist == 0: dist, dx, dy = 1, 1, 0
                             else: dx, dy = (c1.x-c2.x)/dist, (c1.y-c2.y)/dist
                             c1.x += dx*pen*0.5; c1.y += dy*pen*0.5
                             c2.x -= dx*pen*0.5; c2.y -= dy*pen*0.5
            
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
                            nc = Cell(c.x, c.y, split_mass, c.color)
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
                        self.ejected_mass.append(EjectedMass(c.x+math.cos(ang)*c.radius, c.y+math.sin(ang)*c.radius, ang, c.color, p.id, p.team_id))
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
                         pmass = c.mass / (pieces + 1)
                         c.mass = pmass; c.set_recombine_cooldown()
                         for i in range(pieces):
                             ang = (i/pieces)*math.pi*2
                             nc = Cell(c.x, c.y, pmass, c.color)
                             nc.apply_force(math.cos(ang)*SPLIT_IMPULSE, math.sin(ang)*SPLIT_IMPULSE)
                             p.cells.append(nc)
            elif t == GameEvent.MERGE_CELLS:
                p = e['player']
                try:
                    c1, c2 = p.cells[e['idx1']], p.cells[e['idx2']]
                    if c1.mass > 0 and c2.mass > 0:
                        c1.mass += c2.mass
                        c1.x = (c1.x + c2.x)/2
                        c1.y = (c1.y + c2.y)/2
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
                        ang = math.atan2(ej.vy, ej.vx)
                        self.viruses.append(Virus(v.x+math.cos(ang)*v.radius*2, v.y+math.sin(ang)*v.radius*2, VIRUS_START_MASS, ang, VIRUS_SHOT_IMPULSE))

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
            if any((c.x-cx)**2 + (c.y-cy)**2 < v_rad_sq for c in p.cells):
                visible_players.append({'id': p.id, 'name': p.name, 'dead': p.is_dead, 'cells': [{'x': int(c.x), 'y': int(c.y), 'm': int(c.mass), 'c': c.color} for c in p.cells]})
        
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
        
        lb = [{'id': p.id, 'name': p.name, 'mass': int(p.total_mass)} for p in sorted([p for p in self.players.values() if not p.is_dead and not p.is_spectator], key=lambda x: x.total_mass, reverse=True)[:10]]

        return {'players': visible_players, 'food': visible_food, 'viruses': visible_viruses, 'ejected': visible_ejected, 'leaderboard': lb}

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
            print(f"Map size updated to {w}x{h}")
        except ValueError: print("Usage: setsize <width> <height>")

    elif cmd[0] == "clearfood":
        world.food = []
        world.food_grid = {}
        print("All food cleared.")

    elif cmd[0] == "foodcfg" and len(cmd) == 3:
        try:
            max_f, rate = int(cmd[1]), int(cmd[2])
            world.max_food = max_f
            print(f"Food Config Updated: Max={max_f}")
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

world = GameWorld()
pid_counter = 0

async def handler(ws):
    global pid_counter
    pid = pid_counter; pid_counter += 1
    player = None
    try:
        async for msg in ws:
            data = json.loads(msg)
            if data['type'] == 'ping': await ws.send(json.dumps({'type':'pong', 'server_name':SERVER_NAME, 'players':len(world.players), 'max_players':MAX_PLAYERS}))
            elif data['type'] == 'join':
                player = Player(ws, pid, data.get('name', 'Guest')[:15])
                world.players[pid] = player
                await ws.send(json.dumps({'type':'init', 'id':pid, 'map':{'w':MAP_WIDTH, 'h':MAP_HEIGHT}}))
            elif data['type'] == 'spectate':
                player = Player(ws, pid, "Spectator", spectate=True)
                world.players[pid] = player
                await ws.send(json.dumps({'type':'init', 'id':pid, 'map':{'w':MAP_WIDTH, 'h':MAP_HEIGHT}}))
            elif player and not player.is_spectator:
                if data['type'] == 'input': 
                    # ★★★ 防呆：檢查輸入數據，如果為 NaN 或 None 則忽略 ★★★
                    tx, ty = data.get('x'), data.get('y')
                    if tx is not None and ty is not None and not math.isnan(tx) and not math.isnan(ty):
                        player.mouse_x, player.mouse_y = tx, ty
                elif data['type'] == 'split': world.event_queue.append({'type': GameEvent.SPLIT_CELL, 'player': player})
                elif data['type'] == 'eject': world.event_queue.append({'type': GameEvent.EJECT_MASS, 'player': player})
    except: pass
    finally: 
        if pid in world.players: del world.players[pid]

async def game_loop():
    while True:
        t1 = time.time()
        world.update()
        for pid, p in world.players.items():
            if isinstance(p, Bot): continue
            try: await p.ws.send(json.dumps({'type':'update', 'data': world.get_view_state(p)}))
            except: pass
        await asyncio.sleep(max(0, TICK_LEN - (time.time()-t1)))

async def main():
    print(f"Running {SERVER_NAME} on {SERVER_PORT}")
    try:
        async with aiohttp.ClientSession() as s: await s.post(f"{MASTER_URL}/register", json={'url':MY_URL, 'name':SERVER_NAME, 'max_players':MAX_PLAYERS})
    except: pass
    await asyncio.gather(websockets.serve(handler, SERVER_HOST, SERVER_PORT), game_loop(), input_loop())

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: pass