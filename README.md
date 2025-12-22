# Agar

一個使用 Python + WebSocket 實作的 Agar.io 遊戲複製版，包含 AI Bot（遺傳演算法驅動）和動態地圖縮放功能。

## 🎮 功能特色

- **多人對戰** - 透過 WebSocket 即時同步
- **AI Bot** - 內建遺傳演算法驅動的智慧 AI 對手
- **動態縮放** - 地圖大小會根據玩家數量自動調整
- **經典玩法**：
  - 吃食物變大 🍕
  - 吃掉比自己小的玩家 👹
  - 撞到病毒會爆開 🦠
  - 分裂攻擊 (`Space`) 和噴射質量 (`W`)
- **觀戰模式** - 可以旁觀其他玩家

## 📁 專案結構

```
agar/
├── server/                # 遊戲伺服器
│   ├── __init__.py
│   ├── config.py          # 設定管理
│   ├── entities.py        # 遊戲實體 (Cell, Player, Virus...)
│   ├── events.py          # 事件類型定義
│   ├── bot.py             # AI Bot 與遺傳演算法
│   ├── game_world.py      # 遊戲世界邏輯
│   ├── commands.py        # 伺服器命令處理
│   └── main.py            # 伺服器入口點
│
├── master/                # Master Server (伺服器列表管理)
│   ├── __init__.py
│   ├── config.py          # Master 設定
│   └── main.py            # Master 入口點
│
├── client/                # 遊戲客戶端
│   ├── index.html         # 主頁面
│   ├── css/
│   │   └── style.css      # 樣式
│   └── js/
│       ├── config.js      # 客戶端設定
│       ├── state.js       # 遊戲狀態管理
│       ├── network.js     # 網路連線
│       ├── renderer.js    # 渲染器
│       ├── input.js       # 輸入處理
│       └── game.js        # 主遊戲控制器
│
├── env.py                  # 環境變數管理模組
├── .env.template           # 環境變數範例
├── game_config.json        # 遊戲參數設定檔
├── pyproject.toml          # Python 專案設定
├── run_server.py           # 啟動遊戲伺服器
├── run_master.py           # 啟動 Master Server
└── README.md
```

## 🚀 快速開始

### 安裝依賴

```bash
# 使用 pip
pip install -e .

# 或安裝開發依賴
pip install -e ".[dev]"
```

### 設定環境變數

```bash
# 複製環境變數範例
cp .env.template .env

# 編輯 .env 設定你的環境
```

### 啟動伺服器

```bash
# 1. 啟動 Master Server（管理伺服器列表）
python run_master.py

# 2. 啟動遊戲伺服器（另開終端）
python run_server.py

# 3. 用瀏覽器開啟 client/index.html 即可遊玩
```

或使用模組方式：

```bash
python -m master.main  # 啟動 Master Server
python -m server.main  # 啟動遊戲伺服器
```

## ⚙️ 設定說明

### 環境變數 (.env)

| 變數名稱 | 預設值 | 說明 |
|---------|-------|------|
| `SERVER_NAME` | `Agar.io AI Lab (Auto-Scale)` | 伺服器名稱 |
| `SERVER_HOST` | `localhost` | 遊戲伺服器主機 |
| `SERVER_PORT` | `8765` | 遊戲伺服器埠號 |
| `MAX_PLAYERS` | `50` | 最大玩家數 |
| `MASTER_HOST` | `localhost` | Master 伺服器主機 |
| `MASTER_PORT` | `8080` | Master 伺服器埠號 |
| `HEALTH_CHECK_INTERVAL` | `30` | 健康檢查間隔 (秒) |
| `OFFLINE_TIMEOUT` | `300` | 離線超時刪除 (秒) |
| `DEBUG` | `false` | 除錯模式 |

### game_config.json

```json
{
    "map_width": 6000,           // 地圖寬度
    "map_height": 6000,          // 地圖高度
    "player_start_mass": 20,     // 玩家初始質量
    "virus_count": 30,           // 病毒數量
    "food_max_count": 1200,      // 最大食物數量
    "dynamic_scaling_enabled": true,  // 啟用動態縮放
    "scaling_player_step": 5,    // 每 N 個玩家觸發縮放
    "scaling_size_percent": 0.2  // 縮放比例
}
```

## 🎮 遊戲控制

| 按鍵 | 功能 |
|------|------|
| 滑鼠移動 | 控制方向 |
| `Space` | 分裂攻擊 |
| `W` | 噴射質量 |
| `ESC` | 開啟/關閉選單 |
| `Q` | 觀戰模式切換追蹤 |
| 滾輪 | 縮放視野 |

## 🛠️ 伺服器命令

在遊戲伺服器終端輸入：

| 命令 | 說明 |
|------|------|
| `reload` | 重新載入設定檔 |
| `setsize <w> <h>` | 設定地圖大小 |
| `addbot [n]` | 新增 n 個 Bot |
| `removebot [n]` | 移除 n 個 Bot |
| `stats` | 顯示演化統計 |
| `find <name>` | 搜尋玩家 |
| `addmass <id> <amount>` | 增加玩家質量 |
| `kill <id>` | 殺死玩家 |
| `killbotall` | 殺死所有 Bot |


