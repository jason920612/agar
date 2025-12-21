"""
WebSocket 處理與主伺服器入口
"""
import asyncio
import json
import math
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import aiohttp
import websockets

from .config import config
from .entities import Player
from .events import GameEvent
from .game_world import GameWorld
from .commands import CommandHandler
from .bot import Bot


class GameServer:
    """遊戲伺服器"""
    
    def __init__(self):
        self.world = GameWorld()
        self.command_handler = CommandHandler(self.world)
    
    async def handle_connection(self, ws):
        """處理 WebSocket 連線"""
        pid = self.world.next_pid()
        player = None
        
        remote_ip = "Unknown"
        if ws.remote_address:
            remote_ip = f"{ws.remote_address[0]}:{ws.remote_address[1]}"
        
        try:
            async for msg in ws:
                data = json.loads(msg)
                msg_type = data.get('type')
                
                if msg_type == 'ping':
                    await ws.send(json.dumps({
                        'type': 'pong',
                        'server_name': config.SERVER_NAME,
                        'players': len(self.world.players),
                        'max_players': config.MAX_PLAYERS
                    }))
                
                elif msg_type == 'join':
                    player = Player(
                        ws, pid, 
                        data.get('name', 'Guest')[:15],
                        ip=remote_ip,
                        map_size=(self.world.map_width, self.world.map_height)
                    )
                    self.world.players[pid] = player
                    print(f"[Join] ID:{pid} Name:{player.name} IP:{remote_ip}")
                    
                    await ws.send(json.dumps({
                        'type': 'init',
                        'id': pid,
                        'map': {'w': self.world.map_width, 'h': self.world.map_height}
                    }))
                
                elif msg_type == 'spectate':
                    player = Player(
                        ws, pid, "Spectator",
                        ip=remote_ip,
                        spectate=True,
                        map_size=(self.world.map_width, self.world.map_height)
                    )
                    self.world.players[pid] = player
                    
                    await ws.send(json.dumps({
                        'type': 'init',
                        'id': pid,
                        'map': {'w': self.world.map_width, 'h': self.world.map_height}
                    }))
                
                elif player and not player.is_spectator:
                    if msg_type == 'input':
                        tx, ty = data.get('x'), data.get('y')
                        if tx is not None and ty is not None:
                            if not math.isnan(tx) and not math.isnan(ty):
                                player.mouse_x, player.mouse_y = tx, ty
                    
                    elif msg_type == 'split':
                        self.world.event_queue.append({
                            'type': GameEvent.SPLIT_CELL,
                            'player': player
                        })
                    
                    elif msg_type == 'eject':
                        self.world.event_queue.append({
                            'type': GameEvent.EJECT_MASS,
                            'player': player
                        })
                
                elif player and player.is_spectator:
                    if msg_type == 'input':
                        tx, ty = data.get('x'), data.get('y')
                        if tx is not None and ty is not None:
                            if not math.isnan(tx) and not math.isnan(ty):
                                player.mouse_x, player.mouse_y = tx, ty
        
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception:
            traceback.print_exc()
        finally:
            if pid in self.world.players:
                del self.world.players[pid]
    
    async def game_loop(self):
        """遊戲主迴圈"""
        print("Game loop started (Auto-Scale Mode).")
        
        while True:
            t1 = time.time()
            
            try:
                self.world.update()
            except Exception:
                print("!!! CRITICAL ERROR IN GAME LOOP !!!")
                traceback.print_exc()
                await asyncio.sleep(1)
            
            active_players_snapshot = list(self.world.players.items())
            
            # 廣播地圖大小變更
            if self.world.map_needs_sync:
                map_data = json.dumps({
                    'type': 'map_update',
                    'map': {'w': self.world.map_width, 'h': self.world.map_height}
                })
                
                for pid, p in active_players_snapshot:
                    if isinstance(p, Bot):
                        continue
                    try:
                        await p.ws.send(map_data)
                    except:
                        pass
                
                self.world.map_needs_sync = False
            
            # 廣播遊戲狀態
            for pid, p in active_players_snapshot:
                if isinstance(p, Bot):
                    continue
                
                try:
                    if p.is_dead and not p.is_spectator:
                        await p.ws.send(json.dumps({'type': 'death'}))
                    else:
                        await p.ws.send(json.dumps({
                            'type': 'update',
                            'data': self.world.get_view_state(p)
                        }))
                except websockets.exceptions.ConnectionClosed:
                    pass
                except Exception as e:
                    print(f"Send Error: {e}")
            
            process_time = time.time() - t1
            delay = config.TICK_LEN - process_time
            await asyncio.sleep(max(0.001, delay))
    
    async def input_loop(self):
        """命令輸入迴圈"""
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor(1, "InputThread") as executor:
            while True:
                cmd = await loop.run_in_executor(executor, sys.stdin.readline)
                cmd = cmd.strip().split()
                if cmd:
                    self.command_handler.handle(cmd)
    
    async def run(self):
        """啟動伺服器"""
        print(f"Running {config.SERVER_NAME} on {config.SERVER_PORT}")
        
        # 向 Master Server 註冊
        my_url = f"ws://{config.SERVER_HOST}:{config.SERVER_PORT}"
        try:
            async with aiohttp.ClientSession() as session:
                await session.post(
                    f"{config.MASTER_URL}/register",
                    json={
                        'url': my_url,
                        'name': config.SERVER_NAME,
                        'max_players': config.MAX_PLAYERS
                    }
                )
        except Exception:
            print("Warning: Could not register with Master Server")
        
        # 啟動 WebSocket 伺服器
        server = websockets.serve(
            self.handle_connection,
            config.SERVER_HOST,
            config.SERVER_PORT,
            ping_interval=None,
            ping_timeout=None
        )
        
        await asyncio.gather(server, self.game_loop(), self.input_loop())


def main():
    """主函式"""
    server = GameServer()
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\nServer shutting down...")


if __name__ == "__main__":
    main()
