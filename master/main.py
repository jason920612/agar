"""
Master Server - 伺服器列表管理中心
負責管理遊戲伺服器的註冊、健康檢查與列表提供
"""
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any

from aiohttp import web
import websockets

from .config import CHECK_INTERVAL, OFFLINE_TIMEOUT, LISTEN_HOST, LISTEN_PORT

# 取得專案根目錄與 client 目錄路徑
PROJECT_ROOT = Path(__file__).parent.parent
CLIENT_DIR = PROJECT_ROOT / "client"


class MasterServer:
    """Master Server 類別"""
    
    def __init__(self):
        # 儲存格式: { "url": { "name": str, "max": int, "last_seen": time, 
        #                     "offline_since": time/None, "url": str } }
        self.servers: Dict[str, Dict[str, Any]] = {}
    
    async def handle_list(self, request: web.Request) -> web.Response:
        """提供給前端的伺服器列表 API"""
        active_servers = []
        
        for key, srv in self.servers.items():
            status = "online" if srv['offline_since'] is None else "offline"
            
            active_servers.append({
                'url': srv['url'],
                'name': srv['name'],
                'max_players': srv['max'],
                'status': status
            })
        
        return web.json_response(active_servers, headers={
            'Access-Control-Allow-Origin': '*'
        })
    
    async def handle_register(self, request: web.Request) -> web.Response:
        """遊戲伺服器註冊 API"""
        try:
            data = await request.json()
            url = data.get('url')
            name = data.get('name')
            max_p = data.get('max_players')
            
            if url not in self.servers:
                print(f"[Register] New Server: {name} ({url})")
            else:
                print(f"[Heartbeat] Update: {name}")
            
            self.servers[url] = {
                'url': url,
                'name': name,
                'max': max_p,
                'last_seen': time.time(),
                'offline_since': None
            }
            
            return web.Response(text="Registered")
        except Exception as e:
            return web.Response(status=400, text=str(e))
    
    async def handle_deregister(self, request: web.Request) -> web.Response:
        """遊戲伺服器主動關閉 API"""
        try:
            data = await request.json()
            url = data.get('url')
            
            if url in self.servers:
                del self.servers[url]
                print(f"[Deregister] Server removed: {url}")
            
            return web.Response(text="Deregistered")
        except Exception:
            return web.Response(status=400)
    
    async def health_check_task(self):
        """後台任務：定期檢查伺服器狀態"""
        while True:
            print(f"--- Starting Health Check ({len(self.servers)} servers) ---")
            current_time = time.time()
            to_remove = []
            
            for url, srv in self.servers.items():
                try:
                    async with websockets.connect(url, open_timeout=3) as ws:
                        await ws.send(json.dumps({'type': 'ping'}))
                        response = await asyncio.wait_for(ws.recv(), timeout=3)
                        data = json.loads(response)
                        
                        if data['type'] == 'pong':
                            srv['last_seen'] = current_time
                            srv['offline_since'] = None
                            # 可以更新玩家數量
                            # srv['current_players'] = data.get('players', 0)
                
                except Exception:
                    if srv['offline_since'] is None:
                        srv['offline_since'] = current_time
                        print(f"[Warning] Server {srv['name']} is unreachable.")
                    
                    if current_time - srv['offline_since'] > OFFLINE_TIMEOUT:
                        print(f"[Timeout] Removing server {srv['name']} (Offline > {OFFLINE_TIMEOUT}s)")
                        to_remove.append(url)
            
            for url in to_remove:
                del self.servers[url]
            
            await asyncio.sleep(CHECK_INTERVAL)
    
    def create_app(self) -> web.Application:
        """建立 Web 應用"""
        app = web.Application()
        
        # API 路由
        app.add_routes([
            web.get('/list', self.handle_list),
            web.post('/register', self.handle_register),
            web.post('/deregister', self.handle_deregister)
        ])
        
        # 靜態檔案服務 (client 目錄)
        # 根路徑 "/" 返回 index.html
        async def index_handler(request):
            return web.FileResponse(CLIENT_DIR / "index.html")
        
        app.router.add_get('/', index_handler)
        
        # 提供 client 子目錄的靜態檔案 (css, js 等)
        app.router.add_static('/css/', CLIENT_DIR / "css", name='css')
        app.router.add_static('/js/', CLIENT_DIR / "js", name='js')
        
        return app
    
    async def run(self):
        """啟動 Master Server"""
        app = self.create_app()
        runner = web.AppRunner(app)
        await runner.setup()
        
        site = web.TCPSite(runner, LISTEN_HOST, LISTEN_PORT)
        
        print(f"Master Server running on http://{LISTEN_HOST}:{LISTEN_PORT}")
        
        await asyncio.gather(
            site.start(),
            self.health_check_task()
        )
        
        # 保持運行
        await asyncio.Future()


def main():
    """主函式"""
    server = MasterServer()
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\nMaster Server shutting down...")


if __name__ == "__main__":
    main()
