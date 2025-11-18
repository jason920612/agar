import asyncio
import json
import math
import random
import time
import websockets
import sys
import aiohttp # 需要 pip install aiohttp
from concurrent.futures import ThreadPoolExecutor

# --- 伺服器設定 ---
SERVER_NAME = "Taiwan No.1 Server"
SERVER_HOST = "localhost" # 本機對外 IP 或域名
SERVER_PORT = 8765
MAX_PLAYERS = 50
# ★★★ 中央伺服器地址 ★★★
MASTER_URL = "http://localhost:8080" 
MY_URL = f"ws://{SERVER_HOST}:{SERVER_PORT}"

# --- 遊戲常數 ---
MAP_WIDTH = 3000
MAP_HEIGHT = 3000
TICK_RATE = 20
TICK_LEN = 1 / TICK_RATE
MASS_DECAY_RATE = 0.01 

# --- 物理與平衡參數 ---
BASE_MASS = 20
MAX_CELLS = 16
SPLIT_IMPULSE = 750
EJECT_IMPULSE = 500
FRICTION = 0.92
VIRUS_START_MASS = 100
VIRUS_MAX_MASS = 180
VIRUS_COUNT = 30
VIRUS_SHOT_IMPULSE = 800

def mass_to_radius(mass):
    return 6 * math.sqrt(mass)

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

class EjectedMass:
    def __init__(self, x, y, angle, color, parent_id):
        self.id = random.randint(100000, 999999)
        self.x = x
        self.y = y
        self.mass = 15
        self.color = color
        self.radius = mass_to_radius(self.mass)
        self.vx = math.cos(angle) * EJECT_IMPULSE
        self.vy = math.sin(angle) * EJECT_IMPULSE
        self.parent_id = parent_id
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
            cell.mass -= 18
            dx = self.mouse_x - cell.x
            dy = self.mouse_y - cell.y
            angle = math.atan2(dy, dx)
            
            start_x = cell.x + math.cos(angle) * cell.radius
            start_y = cell.y + math.sin(angle) * cell.radius
            ejected = EjectedMass(start_x, start_y, angle, cell.color, self.id)
            world.ejected_mass.append(ejected)

    def explode_on_virus(self, cell_index, virus_mass):
        target_cell = self.cells[cell_index]
        
        # 總是增加病毒的質量
        target_cell.mass += virus_mass
        
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
        name = f"Bot-{chr(random.randint(65, 90))}{random.randint(10,99)}"
        super().__init__(None, pid, name)
        self.genes = {
            'food_weight': random.uniform(1, 8),
            'prey_weight': random.uniform(5, 25),
            'predator_weight': random.uniform(10, 50),
            'virus_weight': random.uniform(-5, 30),
            'split_aggression': random.uniform(0, 1)
        }
        self.color = "#%06x" % random.randint(0xAAAAAA, 0xFFFFFF)

    def think(self, world):
        if self.is_dead or not self.cells:
            if self.is_dead: self.spawn()
            return

        cx = sum(c.x for c in self.cells) / len(self.cells)
        cy = sum(c.y for c in self.cells) / len(self.cells)
        avg_mass = self.total_mass / len(self.cells)

        move_x, move_y = 0, 0
        
        # 1. Food
        closest_food = None
        min_dist_sq = float('inf')
        view_range_sq = 800**2
        
        for f in world.food:
            d_sq = (f['x'] - cx)**2 + (f['y'] - cy)**2
            if d_sq < view_range_sq and d_sq < min_dist_sq:
                min_dist_sq = d_sq
                closest_food = f
        
        if closest_food:
            dx = closest_food['x'] - cx
            dy = closest_food['y'] - cy
            move_x += dx * self.genes['food_weight']
            move_y += dy * self.genes['food_weight']

        # 2. Players
        nearest_prey_dist = float('inf')
        for pid, p in world.players.items():
            if pid == self.id or p.is_dead or p.is_spectator: continue
            for cell in p.cells:
                dx = cell.x - cx
                dy = cell.y - cy
                dist_sq = dx**2 + dy**2
                dist = math.sqrt(dist_sq)
                if dist > 1000: continue 
                ratio = cell.mass / avg_mass
                if ratio > 1.25:
                    force = self.genes['predator_weight'] / (dist + 1) * 5000
                    move_x -= (dx/dist) * force
                    move_y -= (dy/dist) * force
                elif avg_mass > cell.mass * 1.25:
                    if dist < nearest_prey_dist: nearest_prey_dist = dist
                    force = self.genes['prey_weight'] / (dist + 1) * 3000
                    move_x += (dx/dist) * force
                    move_y += (dy/dist) * force

        for v in world.viruses:
            dx = v.x - cx
            dy = v.y - cy
            dist_sq = dx**2 + dy**2
            if dist_sq < 600**2:
                dist = math.sqrt(dist_sq)
                if avg_mass > v.mass * 1.1:
                    force = self.genes['virus_weight'] / (dist + 1) * 5000
                    move_x -= (dx/dist) * force
                    move_y -= (dy/dist) * force

        edge_dist = 200
        if cx < edge_dist: move_x += 500
        if cx > MAP_WIDTH - edge_dist: move_x -= 500
        if cy < edge_dist: move_y += 500
        if cy > MAP_HEIGHT - edge_dist: move_y -= 500

        self.mouse_x = cx + move_x
        self.mouse_y = cy + move_y

        if (nearest_prey_dist < 400 and self.total_mass > 200 and 
            len(self.cells) < MAX_CELLS and random.random() < (0.02 * self.genes['split_aggression'])):
            self.split()

class GameWorld:
    def __init__(self):
        self.players = {}
        self.food = []
        self.viruses = []
        self.ejected_mass = []
        self.max_food = 500
        self.food_spawn_rate = 5
        self.events = []
        self.generate_food(200)
        self.generate_viruses(VIRUS_COUNT)

    def generate_food(self, amount):
        for _ in range(amount):
            self.food.append({
                'id': random.randint(0, 1000000),
                'x': random.randint(0, MAP_WIDTH),
                'y': random.randint(0, MAP_HEIGHT),
                'color': "#%06x" % random.randint(0, 0xFFFFFF),
                'mass': random.randint(2, 5)
            })
            
    def generate_viruses(self, amount):
        for _ in range(amount):
            self.viruses.append(Virus(random.randint(50, MAP_WIDTH-50), random.randint(50, MAP_HEIGHT-50)))

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
                            force = 20 * TICK_LEN
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
                        force = penetration * 0.5
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
                    if e.parent_id == p.id and (now - e.birth_time < 0.5): continue
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
                    for cell2 in p2.cells:
                        dist = math.sqrt((cell1.x - cell2.x)**2 + (cell1.y - cell2.y)**2)
                        if dist < cell1.radius and cell1.mass > cell2.mass * 1.25:
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

# --- 中央伺服器通訊函數 ---
async def register_to_master():
    """向中央伺服器註冊"""
    try:
        async with aiohttp.ClientSession() as session:
            data = {
                'url': MY_URL,
                'name': SERVER_NAME,
                'max_players': MAX_PLAYERS
            }
            async with session.post(f"{MASTER_URL}/register", json=data) as resp:
                if resp.status == 200:
                    print("Successfully registered to Master Server")
                else:
                    print(f"Failed to register: {resp.status}")
    except Exception as e:
        print(f"Master Server Error: {e}")

async def deregister_from_master():
    """向中央伺服器註銷"""
    try:
        async with aiohttp.ClientSession() as session:
            data = {'url': MY_URL}
            async with session.post(f"{MASTER_URL}/deregister", json=data) as resp:
                print("Deregistered from Master Server")
    except:
        pass

def manage_game_commands(cmd):
    global pid_counter, MAP_WIDTH, MAP_HEIGHT
    
    if cmd[0] == "addbot" and len(cmd) > 1:
        try:
            n = int(cmd[1])
            print(f"Adding {n} bots...")
            for _ in range(n):
                pid = pid_counter
                pid_counter += 1
                world.players[pid] = Bot(pid)
        except ValueError: print("Invalid number")
        
    elif cmd[0] == "removebot" and len(cmd) > 1:
        try:
            n = int(cmd[1])
            print(f"Removing {n} bots...")
            bots_to_remove = [pid for pid, p in world.players.items() if isinstance(p, Bot)]
            for pid in bots_to_remove[:n]: del world.players[pid]
        except ValueError: print("Invalid number")

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
        print("Commands: addbot, removebot, setsize, clearfood, foodcfg")

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
            
            # 響應 Master 的 Ping
            if data['type'] == 'ping':
                player_count = len([p for p in world.players.values() if not isinstance(p, Bot)])
                await websocket.send(json.dumps({
                    'type': 'pong', 
                    'server_name': SERVER_NAME,
                    'players': player_count,
                    'max_players': MAX_PLAYERS
                }))
                
            elif data['type'] == 'join':
                name = data.get('name', 'Guest')[:15]
                current_player = Player(websocket, pid, name)
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
                elif data['type'] == 'split':
                    current_player.split()
                elif data['type'] == 'eject':
                    current_player.eject(world)
                elif data['type'] == 'respawn' and current_player.is_dead:
                    current_player.spawn()
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

        websockets_to_remove = []
        for pid, p in world.players.items():
            if isinstance(p, Bot): continue
            try: 
                await p.websocket.send(msg_state)
                for event_msg in msg_events:
                    await p.websocket.send(event_msg)
            except: websockets_to_remove.append(pid)
        
        for pid in websockets_to_remove: del world.players[pid]
        await asyncio.sleep(max(0, TICK_LEN - (time.time() - start)))

async def main():
    print(f"Agar.io Game Server v4.1 - {SERVER_NAME}")
    
    # 啟動時註冊
    await register_to_master()
    
    try:
        await asyncio.gather(game_loop(), websockets.serve(handler, SERVER_HOST, SERVER_PORT), input_loop())
    finally:
        # 結束時註銷
        await deregister_from_master()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # 強制註銷 (盡力而為)
        asyncio.run(deregister_from_master())