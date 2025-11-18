import asyncio
import json
import time
from aiohttp import web
import websockets

# --- 中央伺服器設定 ---
CHECK_INTERVAL = 30      # 每 30 秒檢查一次
OFFLINE_TIMEOUT = 300    # 離線超過 300 秒刪除
LISTEN_PORT = 8080       # Master Server 監聽的 Port

# 儲存格式: { "ip:port": { "name":Str, "max":Int, "last_seen":Time, "offline_since":Time/None, "url":Str } }
servers = {}

async def handle_list(request):
    """提供給前端的列表 API"""
    # 只回傳在線或離線時間未達刪除標準的伺服器
    active_servers = []
    for key, srv in servers.items():
        status = "online"
        if srv['offline_since']:
            status = "offline"
        
        active_servers.append({
            'url': srv['url'],
            'name': srv['name'],
            'max_players': srv['max'],
            'status': status
        })
    
    return web.json_response(active_servers, headers={
        'Access-Control-Allow-Origin': '*'  # 允許跨域請求
    })

async def handle_register(request):
    """遊戲伺服器註冊 API"""
    try:
        data = await request.json()
        url = data.get('url') # e.g., "ws://localhost:8765"
        name = data.get('name')
        max_p = data.get('max_players')
        
        if url not in servers:
            print(f"[Register] New Server: {name} ({url})")
        else:
            print(f"[Heartbeat] Update: {name}")

        servers[url] = {
            'url': url,
            'name': name,
            'max': max_p,
            'last_seen': time.time(),
            'offline_since': None # 重置離線狀態
        }
        return web.Response(text="Registered")
    except Exception as e:
        return web.Response(status=400, text=str(e))

async def handle_deregister(request):
    """遊戲伺服器主動關閉 API"""
    try:
        data = await request.json()
        url = data.get('url')
        if url in servers:
            del servers[url]
            print(f"[Deregister] Server removed: {url}")
        return web.Response(text="Deregistered")
    except:
        return web.Response(status=400)

async def health_check_task():
    """後台任務：定期檢查伺服器狀態"""
    while True:
        print(f"--- Starting Health Check ({len(servers)} servers) ---")
        current_time = time.time()
        to_remove = []

        for url, srv in servers.items():
            # 嘗試連線測試
            try:
                async with websockets.connect(url, open_timeout=3) as ws:
                    # 發送 ping 請求
                    await ws.send(json.dumps({'type': 'ping'}))
                    response = await asyncio.wait_for(ws.recv(), timeout=3)
                    data = json.loads(response)
                    
                    if data['type'] == 'pong':
                        # 連線成功
                        srv['last_seen'] = current_time
                        srv['offline_since'] = None
                        # 可以順便更新人數資訊 (可選)
                        # srv['current_players'] = data['players'] 
            except Exception as e:
                # 連線失敗
                if srv['offline_since'] is None:
                    srv['offline_since'] = current_time
                    print(f"[Warning] Server {srv['name']} is unreachable.")
                
                # 檢查是否超時
                if current_time - srv['offline_since'] > OFFLINE_TIMEOUT:
                    print(f"[Timeout] Removing server {srv['name']} (Offline > 300s)")
                    to_remove.append(url)

        for url in to_remove:
            del servers[url]
            
        await asyncio.sleep(CHECK_INTERVAL)

async def main():
    app = web.Application()
    app.add_routes([
        web.get('/list', handle_list),
        web.post('/register', handle_register),
        web.post('/deregister', handle_deregister)
    ])
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', LISTEN_PORT)
    
    print(f"Master Server running on http://localhost:{LISTEN_PORT}")
    
    # 同時運行網頁伺服器和健康檢查任務
    await asyncio.gather(
        site.start(),
        health_check_task()
    )
    # 保持運行
    await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass