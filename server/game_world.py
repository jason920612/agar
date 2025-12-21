"""
遊戲世界模組
管理遊戲狀態、物理更新、碰撞偵測
"""
import math
import random
import time
from typing import Dict, List, Any, Optional

from .config import config
from .entities import Cell, Virus, EjectedMass, Player, mass_to_radius
from .events import GameEvent
from .bot import Bot, evo_manager


class GameWorld:
    """遊戲世界管理類"""
    
    def __init__(self):
        # 玩家與實體
        self.players: Dict[int, Player] = {}
        self.food: List[Dict] = []
        self.food_grid: Dict[tuple, List[Dict]] = {}
        self.viruses: List[Virus] = []
        self.ejected_mass: List[EjectedMass] = []
        self.event_queue: List[Dict] = []
        
        # 動態縮放狀態
        self.map_width = config.map_width
        self.map_height = config.map_height
        self.max_food = config.food_max_count
        self.target_virus_count = config.virus_count
        
        # 基準值（用於動態縮放計算）
        self.base_map_width = config.map_width
        self.base_map_height = config.map_height
        self.base_food_count = config.food_max_count
        self.base_virus_count = config.virus_count
        
        self.last_scaling_check = 0
        self.map_needs_sync = False
        
        # PID 計數器
        self.pid_counter = 0
        
        # 初始化食物與病毒
        self.generate_food(int(self.max_food * 0.6))
        self.generate_viruses(self.target_virus_count)
    
    def next_pid(self) -> int:
        """取得下一個玩家 ID"""
        pid = self.pid_counter
        self.pid_counter += 1
        return pid
    
    # --- 食物網格管理 ---
    
    def add_food_to_grid(self, f: Dict):
        """將食物加入網格"""
        gx = int(f['x'] // config.GRID_SIZE)
        gy = int(f['y'] // config.GRID_SIZE)
        if (gx, gy) not in self.food_grid:
            self.food_grid[(gx, gy)] = []
        self.food_grid[(gx, gy)].append(f)
    
    def remove_food_from_grid(self, f: Dict):
        """從網格移除食物"""
        gx = int(f['x'] // config.GRID_SIZE)
        gy = int(f['y'] // config.GRID_SIZE)
        if (gx, gy) in self.food_grid:
            try:
                self.food_grid[(gx, gy)].remove(f)
            except ValueError:
                pass
            if not self.food_grid[(gx, gy)]:
                del self.food_grid[(gx, gy)]
    
    # --- 生成器 ---
    
    def generate_food(self, amount: int):
        """生成食物"""
        min_m = config.food_min_mass
        max_m = config.food_max_mass
        
        for _ in range(amount):
            f = {
                'id': random.randint(0, 1_000_000_000),
                'x': random.randint(0, self.map_width),
                'y': random.randint(0, self.map_height),
                'color': "#%06x" % random.randint(0, 0xFFFFFF),
                'mass': random.randint(min_m, max_m)
            }
            self.food.append(f)
            self.add_food_to_grid(f)
    
    def generate_viruses(self, amount: int):
        """生成病毒"""
        for _ in range(amount):
            self.viruses.append(Virus(
                random.randint(100, self.map_width - 100),
                random.randint(100, self.map_height - 100)
            ))
    
    # --- 動態縮放 ---
    
    def check_dynamic_scaling(self):
        """計算並應用動態地圖與資源縮放"""
        if not config.dynamic_scaling_enabled:
            return
        
        player_count = len([p for p in self.players.values() if not p.is_spectator])
        step = config.scaling_player_step
        size_pct = config.scaling_size_percent
        res_pct = config.scaling_resource_percent
        
        multiplier = player_count // step
        
        new_width = int(self.base_map_width * (1 + multiplier * size_pct))
        new_height = int(self.base_map_height * (1 + multiplier * size_pct))
        new_max_food = int(self.base_food_count * (1 + multiplier * res_pct))
        new_virus_count = int(self.base_virus_count * (1 + multiplier * res_pct))
        
        if new_width != self.map_width or new_height != self.map_height:
            print(f"[Scale] Players: {player_count}, Size: {new_width}x{new_height}, "
                  f"Food: {new_max_food}, Virus: {new_virus_count}")
            self.map_width = new_width
            self.map_height = new_height
            self.max_food = new_max_food
            self.target_virus_count = new_virus_count
            self.map_needs_sync = True
    
    # --- 強制分裂 ---
    
    def enforce_max_mass(self):
        """強制分裂過大的細胞"""
        for p in self.players.values():
            if p.is_dead or p.is_spectator:
                continue
            
            cells_to_add = []
            
            for c in p.cells:
                if c.mass >= config.max_cell_mass:
                    split_mass = c.mass / 2
                    c.mass = split_mass
                    c.set_recombine_cooldown()
                    
                    dx, dy = p.mouse_x - c.x, p.mouse_y - c.y
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist == 0:
                        dx, dy = 1, 0
                    else:
                        dx, dy = dx / dist, dy / dist
                    
                    nc = Cell(c.x + dx * c.radius, c.y + dy * c.radius, 
                             split_mass, c.color)
                    c.apply_force(-dx * 100, -dy * 100)
                    nc.apply_force(dx * config.SPLIT_IMPULSE, dy * config.SPLIT_IMPULSE)
                    
                    cells_to_add.append(nc)
            
            if cells_to_add:
                p.cells.extend(cells_to_add)
    
    # --- 主更新迴圈 ---
    
    def update(self):
        """更新遊戲狀態"""
        now = time.time()
        
        # 週期性檢查地圖縮放（每 5 秒）
        if now - self.last_scaling_check > 5:
            self.check_dynamic_scaling()
            self.last_scaling_check = now
        
        # 強制分裂檢查
        self.enforce_max_mass()
        
        # 更新拋射物與病毒位置
        for e in self.ejected_mass:
            e.move(self.map_width, self.map_height)
        for v in self.viruses:
            v.move(self.map_width, self.map_height)
        
        # Bot 更新
        self._update_bots()
        
        # 玩家移動與碰撞
        active_players = [p for p in self.players.values() if not p.is_dead]
        
        # 細胞合併吸引力
        self._apply_merge_attraction(active_players, now)
        
        # 細胞移動
        self._move_cells(active_players)
        
        # 碰撞偵測
        self._detect_collisions(active_players, now)
        
        # 處理事件
        self.process_events()
        
        # 補充食物與病毒
        if len(self.food) < self.max_food:
            self.generate_food(10)
        if len(self.viruses) < self.target_virus_count:
            self.generate_viruses(1)
    
    def _update_bots(self):
        """更新 Bot"""
        for p in list(self.players.values()):
            if isinstance(p, Bot):
                if p.is_dead:
                    evo_manager.record_genome(p)
                    new_genes = evo_manager.get_next_generation_genes()
                    del self.players[p.id]
                    
                    new_id = self.next_pid()
                    new_bot = Bot(new_id, genes=new_genes, 
                                 map_size=(self.map_width, self.map_height))
                    new_bot.spawn()
                    self.players[new_id] = new_bot
                else:
                    action = p.decide(self)
                    if action == 'split':
                        self.event_queue.append({
                            'type': GameEvent.SPLIT_CELL, 
                            'player': p
                        })
                    elif action == 'eject':
                        self.event_queue.append({
                            'type': GameEvent.EJECT_MASS, 
                            'player': p
                        })
    
    def _apply_merge_attraction(self, active_players: List[Player], now: float):
        """套用合併吸引力"""
        for p in active_players:
            if len(p.cells) > 1:
                for i in range(len(p.cells)):
                    for j in range(i + 1, len(p.cells)):
                        c1 = p.cells[i]
                        c2 = p.cells[j]
                        
                        if now > c1.recombine_time and now > c2.recombine_time:
                            dx = c2.x - c1.x
                            dy = c2.y - c1.y
                            dist = math.sqrt(dx**2 + dy**2)
                            
                            if dist > 0:
                                attraction_factor = config.merge_attraction_force
                                safe_mass1 = max(c1.mass, 1)
                                safe_mass2 = max(c2.mass, 1)
                                base_speed_c1 = 300 * (safe_mass1 ** -0.2)
                                base_speed_c2 = 300 * (safe_mass2 ** -0.2)
                                
                                pull_x = (dx / dist) * config.TICK_LEN
                                pull_y = (dy / dist) * config.TICK_LEN
                                
                                c1.x += pull_x * base_speed_c1 * attraction_factor
                                c1.y += pull_y * base_speed_c1 * attraction_factor
                                c2.x -= pull_x * base_speed_c2 * attraction_factor
                                c2.y -= pull_y * base_speed_c2 * attraction_factor
    
    def _move_cells(self, active_players: List[Player]):
        """移動細胞"""
        for p in active_players:
            for cell in p.cells:
                cell.move(p.mouse_x, p.mouse_y, self.map_width, self.map_height)
                cell.decay()
    
    def _detect_collisions(self, active_players: List[Player], now: float):
        """偵測碰撞"""
        # 食物、拋射物、病毒碰撞
        for p in active_players:
            for cell in p.cells:
                self._check_food_collision(cell)
                self._check_ejected_collision(cell, p, now)
                self._check_virus_collision(cell, p)
        
        # 細胞合併與玩家互吃
        self._check_cell_interactions(active_players, now)
        
        # 病毒與拋射物碰撞
        self._check_virus_ejected_collision()
    
    def _check_food_collision(self, cell: Cell):
        """檢查食物碰撞"""
        min_gx = int((cell.x - cell.radius) // config.GRID_SIZE)
        max_gx = int((cell.x + cell.radius) // config.GRID_SIZE)
        min_gy = int((cell.y - cell.radius) // config.GRID_SIZE)
        max_gy = int((cell.y + cell.radius) // config.GRID_SIZE)
        
        for gx in range(min_gx, max_gx + 1):
            for gy in range(min_gy, max_gy + 1):
                if (gx, gy) in self.food_grid:
                    for f in self.food_grid[(gx, gy)]:
                        dx = cell.x - f['x']
                        dy = cell.y - f['y']
                        if dx*dx + dy*dy < cell.radius**2:
                            self.event_queue.append({
                                'type': GameEvent.EAT_FOOD,
                                'cell': cell,
                                'food': f
                            })
    
    def _check_ejected_collision(self, cell: Cell, player: Player, now: float):
        """檢查拋射物碰撞"""
        for e in self.ejected_mass:
            if e.parent_id == player.id and now - e.birth_time < 0.2:
                continue
            dx = cell.x - e.x
            dy = cell.y - e.y
            if dx*dx + dy*dy < cell.radius**2:
                self.event_queue.append({
                    'type': GameEvent.EAT_EJECTED,
                    'cell': cell,
                    'ejected': e
                })
    
    def _check_virus_collision(self, cell: Cell, player: Player):
        """檢查病毒碰撞"""
        for v in self.viruses:
            if cell.mass > v.mass * 1.1:
                dx = cell.x - v.x
                dy = cell.y - v.y
                if dx*dx + dy*dy < cell.radius**2:
                    self.event_queue.append({
                        'type': GameEvent.EAT_VIRUS,
                        'player': player,
                        'cell_idx': player.cells.index(cell),
                        'virus': v
                    })
    
    def _check_cell_interactions(self, active_players: List[Player], now: float):
        """檢查細胞互動（合併、互吃）"""
        for p1 in active_players:
            # 自己細胞的合併
            for i in range(len(p1.cells)):
                for j in range(i + 1, len(p1.cells)):
                    c1, c2 = p1.cells[i], p1.cells[j]
                    dist = math.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)
                    radius_sum = c1.radius + c2.radius
                    
                    if dist < radius_sum:
                        can_merge = now > c1.recombine_time and now > c2.recombine_time
                        
                        if can_merge and dist < radius_sum * 0.65:
                            self.event_queue.append({
                                'type': GameEvent.MERGE_CELLS,
                                'player': p1,
                                'idx1': i,
                                'idx2': j
                            })
                        elif not can_merge:
                            # 推開
                            pen = radius_sum - dist
                            if dist == 0:
                                rand_ang = random.uniform(0, math.pi * 2)
                                dx, dy = math.cos(rand_ang), math.sin(rand_ang)
                            else:
                                dx = (c1.x - c2.x) / dist
                                dy = (c1.y - c2.y) / dist
                            
                            f = 0.5
                            c1.x += dx * pen * f
                            c1.y += dy * pen * f
                            c2.x -= dx * pen * f
                            c2.y -= dy * pen * f
            
            # 與其他玩家的互吃
            for p2 in active_players:
                if p1.id == p2.id:
                    continue
                
                for c1 in p1.cells:
                    for c2 in p2.cells:
                        if c1.mass > c2.mass * 1.25:
                            dist = math.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)
                            if dist < c1.radius:
                                self.event_queue.append({
                                    'type': GameEvent.EAT_PLAYER_CELL,
                                    'predator': c1,
                                    'prey': c2,
                                    'prey_p': p2
                                })
    
    def _check_virus_ejected_collision(self):
        """檢查病毒與拋射物碰撞"""
        for v in self.viruses:
            for e in self.ejected_mass:
                dx = v.x - e.x
                dy = v.y - e.y
                if dx*dx + dy*dy < (v.radius + e.radius)**2:
                    self.event_queue.append({
                        'type': GameEvent.VIRUS_SPLIT,
                        'virus': v,
                        'ejected': e
                    })
    
    # --- 事件處理 ---
    
    def process_events(self):
        """處理事件佇列"""
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
                    if ej in self.ejected_mass:
                        self.ejected_mass.remove(ej)
            
            elif t == GameEvent.EAT_PLAYER_CELL:
                prey = e['prey']
                if prey.mass > 0:
                    e['predator'].mass += prey.mass
                    prey.mass = 0
            
            elif t == GameEvent.SPLIT_CELL:
                self._handle_split(e['player'])
            
            elif t == GameEvent.EJECT_MASS:
                self._handle_eject(e['player'])
            
            elif t == GameEvent.EAT_VIRUS:
                self._handle_eat_virus(e, removed_viruses)
            
            elif t == GameEvent.MERGE_CELLS:
                self._handle_merge(e)
            
            elif t == GameEvent.VIRUS_SPLIT:
                self._handle_virus_split(e, removed_ejected, removed_viruses)
        
        # 清理死亡細胞
        for p in self.players.values():
            p.cells = [c for c in p.cells if c.mass > 0]
            if not p.cells and not p.is_spectator:
                p.is_dead = True
        
        self.event_queue.clear()
    
    def _handle_split(self, player: Player):
        """處理分裂"""
        if len(player.cells) >= config.MAX_CELLS:
            return
        
        new_cells = []
        for c in player.cells:
            if c.mass >= 36 and len(player.cells) + len(new_cells) < config.MAX_CELLS:
                split_mass = c.mass / 2
                c.mass = split_mass
                
                dx = player.mouse_x - c.x
                dy = player.mouse_y - c.y
                ang = math.atan2(dy, dx)
                
                c.apply_force(-math.cos(ang) * 200, -math.sin(ang) * 200)
                
                nc = Cell(
                    c.x + math.cos(ang) * c.radius,
                    c.y + math.sin(ang) * c.radius,
                    split_mass, c.color
                )
                nc.apply_force(
                    math.cos(ang) * config.SPLIT_IMPULSE,
                    math.sin(ang) * config.SPLIT_IMPULSE
                )
                new_cells.append(nc)
        
        player.cells.extend(new_cells)
    
    def _handle_eject(self, player: Player):
        """處理噴射質量"""
        for c in player.cells:
            if c.mass > 36:
                c.mass -= 16
                
                dx = player.mouse_x - c.x
                dy = player.mouse_y - c.y
                ang = math.atan2(dy, dx)
                
                c.apply_force(-math.cos(ang) * 100, -math.sin(ang) * 100)
                
                ej_x = c.x + math.cos(ang) * c.radius
                ej_y = c.y + math.sin(ang) * c.radius
                
                self.ejected_mass.append(EjectedMass(
                    ej_x, ej_y, ang, c.color, player.id, player.team_id
                ))
    
    def _handle_eat_virus(self, e: Dict, removed_viruses: set):
        """處理吃到病毒"""
        v = e['virus']
        if v.id in removed_viruses:
            return
        
        p = e['player']
        c = p.cells[e['cell_idx']]
        c.mass += v.mass
        removed_viruses.add(v.id)
        
        if v in self.viruses:
            self.viruses.remove(v)
        
        if len(p.cells) < config.MAX_CELLS:
            pieces = min(config.MAX_CELLS - len(p.cells), 7)
            if pieces > 0:
                pmass = c.mass / (pieces + 1)
                c.mass = pmass
                c.set_recombine_cooldown()
                
                for i in range(pieces):
                    ang = (i / pieces) * math.pi * 2 + random.uniform(-0.5, 0.5)
                    nc = Cell(
                        c.x + math.cos(ang) * c.radius * 0.8,
                        c.y + math.sin(ang) * c.radius * 0.8,
                        pmass, c.color
                    )
                    nc.apply_force(
                        math.cos(ang) * config.SPLIT_IMPULSE,
                        math.sin(ang) * config.SPLIT_IMPULSE
                    )
                    p.cells.append(nc)
    
    def _handle_merge(self, e: Dict):
        """處理細胞合併"""
        p = e['player']
        try:
            c1, c2 = p.cells[e['idx1']], p.cells[e['idx2']]
            if c1.mass > 0 and c2.mass > 0:
                total = c1.mass + c2.mass
                c1.x = (c1.x * c1.mass + c2.x * c2.mass) / total
                c1.y = (c1.y * c1.mass + c2.y * c2.mass) / total
                c1.mass = total
                c2.mass = 0
        except (IndexError, ZeroDivisionError):
            pass
    
    def _handle_virus_split(self, e: Dict, removed_ejected: set, removed_viruses: set):
        """處理病毒被餵食後分裂"""
        v = e['virus']
        ej = e['ejected']
        
        if ej.id in removed_ejected or v.id in removed_viruses:
            return
        
        v.mass += ej.mass
        removed_ejected.add(ej.id)
        
        if ej in self.ejected_mass:
            self.ejected_mass.remove(ej)
        
        if v.mass >= config.virus_max_mass:
            v.mass = config.virus_start_mass
            shoot_ang = math.atan2(ej.vy, ej.vx)
            
            self.viruses.append(Virus(
                v.x + math.cos(shoot_ang) * v.radius * 2.5,
                v.y + math.sin(shoot_ang) * v.radius * 2.5,
                config.virus_start_mass,
                shoot_ang,
                config.VIRUS_SHOT_IMPULSE
            ))
    
    # --- 視野狀態 ---
    
    def get_view_state(self, player: Player) -> Dict[str, Any]:
        """取得玩家視野內的遊戲狀態"""
        cx, cy = player.center
        v_rad = 2500
        v_rad_sq = v_rad ** 2
        
        # 可見玩家
        visible_players = []
        for p in self.players.values():
            if p.is_dead or p.is_spectator:
                continue
            
            p_cells = []
            in_view = False
            
            for c in p.cells:
                if (c.x - cx)**2 + (c.y - cy)**2 < v_rad_sq:
                    in_view = True
                p_cells.append({
                    'x': int(c.x),
                    'y': int(c.y),
                    'm': int(c.mass),
                    'c': c.color
                })
            
            if in_view:
                visible_players.append({
                    'id': p.id,
                    'name': p.name,
                    'dead': p.is_dead,
                    'cells': p_cells
                })
        
        # 可見食物
        visible_food = []
        min_gx = max(0, int((cx - v_rad) // config.GRID_SIZE))
        max_gx = min(int(self.map_width // config.GRID_SIZE), 
                     int((cx + v_rad) // config.GRID_SIZE))
        min_gy = max(0, int((cy - v_rad) // config.GRID_SIZE))
        max_gy = min(int(self.map_height // config.GRID_SIZE), 
                     int((cy + v_rad) // config.GRID_SIZE))
        
        for gx in range(min_gx, max_gx + 1):
            for gy in range(min_gy, max_gy + 1):
                if (gx, gy) in self.food_grid:
                    for f in self.food_grid[(gx, gy)]:
                        if (f['x'] - cx)**2 + (f['y'] - cy)**2 < v_rad_sq:
                            visible_food.append(f)
        
        # 可見病毒
        visible_viruses = [
            {'x': int(v.x), 'y': int(v.y), 'm': int(v.mass)}
            for v in self.viruses
            if (v.x - cx)**2 + (v.y - cy)**2 < v_rad_sq
        ]
        
        # 可見拋射物
        visible_ejected = [
            {'x': int(e.x), 'y': int(e.y), 'c': e.color}
            for e in self.ejected_mass
            if (e.x - cx)**2 + (e.y - cy)**2 < v_rad_sq
        ]
        
        # 排行榜
        lb = []
        sorted_players = sorted(
            [p for p in self.players.values() if not p.is_dead and not p.is_spectator],
            key=lambda x: x.total_mass,
            reverse=True
        )[:10]
        
        for p in sorted_players:
            pcx, pcy = p.center
            lb.append({
                'id': p.id,
                'name': p.name,
                'mass': int(p.total_mass),
                'x': int(pcx),
                'y': int(pcy)
            })
        
        return {
            'players': visible_players,
            'food': visible_food,
            'viruses': visible_viruses,
            'ejected': visible_ejected,
            'leaderboard': lb
        }
