"""
遊戲伺服器命令處理模組
"""
from typing import List

from .config import config
from .bot import Bot, evo_manager


class CommandHandler:
    """伺服器命令處理器"""
    
    def __init__(self, world):
        self.world = world
    
    def handle(self, cmd: List[str]) -> None:
        """處理命令"""
        if not cmd:
            return
        
        command = cmd[0].lower()
        handler = getattr(self, f'cmd_{command}', None)
        
        if handler:
            handler(cmd)
        else:
            self.show_help()
    
    def show_help(self):
        """顯示幫助"""
        print("Commands: reload, setsize, clearfood, foodcfg, addbot, "
              "removebot, stats, find, addmass, removemass, kill, killbotall")
    
    def cmd_reload(self, cmd: List[str]):
        """重新載入設定"""
        try:
            config.reload()
            self.world.max_food = config.food_max_count
        except Exception as e:
            print(f"Error reloading config: {e}")
    
    def cmd_setsize(self, cmd: List[str]):
        """設定地圖大小"""
        if len(cmd) != 3:
            print("Usage: setsize <width> <height>")
            return
        
        try:
            w, h = int(cmd[1]), int(cmd[2])
            self.world.map_width = w
            self.world.map_height = h
            self.world.map_needs_sync = True
            print(f"Map size updated to {w}x{h}")
        except ValueError:
            print("Usage: setsize <width> <height>")
    
    def cmd_clearfood(self, cmd: List[str]):
        """清除所有食物"""
        self.world.food = []
        self.world.food_grid = {}
        print("All food cleared.")
    
    def cmd_foodcfg(self, cmd: List[str]):
        """設定食物參數"""
        if len(cmd) != 3:
            print("Usage: foodcfg <max_amount> <spawn_rate>")
            return
        
        try:
            max_f = int(cmd[1])
            self.world.max_food = max_f
            print(f"Food Config Updated: Max={max_f}")
        except ValueError:
            print("Usage: foodcfg <max_amount> <spawn_rate>")
    
    def cmd_addbot(self, cmd: List[str]):
        """新增 Bot"""
        try:
            count = int(cmd[1]) if len(cmd) > 1 else 1
            for _ in range(count):
                bot_id = self.world.next_pid()
                bot = Bot(bot_id, genes=None, 
                         map_size=(self.world.map_width, self.world.map_height))
                self.world.players[bot_id] = bot
                print(f"Bot {bot_id} added.")
        except ValueError:
            print("Usage: addbot <count>")
    
    def cmd_removebot(self, cmd: List[str]):
        """移除 Bot"""
        try:
            count = int(cmd[1]) if len(cmd) > 1 else 1
            removed = 0
            bot_ids = [pid for pid, p in self.world.players.items() 
                      if isinstance(p, Bot)]
            
            for i in range(min(count, len(bot_ids))):
                del self.world.players[bot_ids[i]]
                removed += 1
            
            print(f"Removed {removed} bots.")
        except ValueError:
            print("Usage: removebot <count>")
    
    def cmd_stats(self, cmd: List[str]):
        """顯示演化統計"""
        print("--- Evo Stats ---")
        print(f"Best Score: {int(evo_manager.best_score)}")
        print(f"Gene Pool Size: {len(evo_manager.gene_pool)}")
        
        if evo_manager.gene_pool:
            best_gene = evo_manager.gene_pool[0][1]
            print(f"Top Gene (Gen {best_gene['generation']}): "
                  f"Hunt={int(best_gene['w_hunt'])}, Flee={int(best_gene['w_flee'])}")
    
    def cmd_find(self, cmd: List[str]):
        """搜尋玩家"""
        if len(cmd) < 2:
            print("Usage: find <name_fragment>")
            return
        
        target = cmd[1].lower()
        found = False
        
        print(f"{'ID':<6} {'Name':<15} {'Mass':<8} {'Cells':<6} {'IP Address'}")
        print("-" * 55)
        
        for p in self.world.players.values():
            if target in p.name.lower():
                print(f"{p.id:<6} {p.name[:15]:<15} {int(p.total_mass):<8} "
                      f"{len(p.cells):<6} {p.ip}")
                found = True
        
        if not found:
            print("No matches found.")
    
    def cmd_addmass(self, cmd: List[str]):
        """增加玩家質量"""
        if len(cmd) < 3:
            print("Usage: addmass <player_id> <amount>")
            return
        
        try:
            target_id = int(cmd[1])
            amount = int(cmd[2])
            p = self.world.players.get(target_id)
            
            if p and not p.is_dead and len(p.cells) > 0:
                per_cell = amount / len(p.cells)
                for c in p.cells:
                    c.mass += per_cell
                print(f"Added {amount} mass to {p.name} (ID: {target_id}).")
            else:
                print("Player not found or is dead.")
        except ValueError:
            print("Invalid ID or Amount.")
    
    def cmd_removemass(self, cmd: List[str]):
        """減少玩家質量"""
        if len(cmd) < 3:
            print("Usage: removemass <player_id> <amount>")
            return
        
        try:
            target_id = int(cmd[1])
            amount = int(cmd[2])
            p = self.world.players.get(target_id)
            
            if p and not p.is_dead and len(p.cells) > 0:
                per_cell_loss = amount / len(p.cells)
                for c in p.cells:
                    c.mass = max(10, c.mass - per_cell_loss)
                print(f"Removed {amount} mass from {p.name} (ID: {target_id}).")
            else:
                print("Player not found or is dead.")
        except ValueError:
            print("Invalid ID or Amount.")
    
    def cmd_kill(self, cmd: List[str]):
        """殺死玩家"""
        if len(cmd) < 2:
            print("Usage: kill <player_id>")
            return
        
        try:
            target_id = int(cmd[1])
            p = self.world.players.get(target_id)
            
            if p:
                p.is_dead = True
                p.cells = []
                print(f"Killed player {p.name} (ID: {target_id}).")
            else:
                print("Player not found.")
        except ValueError:
            print("Invalid ID.")
    
    def cmd_killbotall(self, cmd: List[str]):
        """殺死所有 Bot"""
        count = 0
        for p in list(self.world.players.values()):
            if isinstance(p, Bot):
                p.is_dead = True
                p.cells = []
                count += 1
        print(f"Killed {count} bots.")
