/**
 * 主遊戲控制器
 */
import CONFIG from './config.js';
import GameState from './state.js';
import NetworkManager from './network.js';
import Renderer from './renderer.js';
import InputHandler from './input.js';

class Game {
    constructor() {
        this.state = new GameState();
        this.network = new NetworkManager(this);
        this.renderer = new Renderer(document.getElementById('gameCanvas'));
        this.input = new InputHandler(this);

        // 玩家狀態
        this.myId = -1;
        this.mapSize = { w: 6000, h: 6000 };
        this.localPlayers = {};

        // 攝影機
        this.camX = 0;
        this.camY = 0;
        this.camScale = 0.5;
        this.targetCamScale = 0.5;

        // 遊戲狀態
        this.isPlaying = false;
        this.isSpectating = false;
        this.isDead = true;
        this.specFollowMode = true;

        // FPS 計數
        this.frameCount = 0;
        this.lastFpsUpdate = Date.now();

        this.init();
    }

    init() {
        // 綁定 UI 事件
        document.getElementById('playBtn').onclick = () => this.play();
        document.getElementById('specBtn').onclick = () => this.spectate();
        document.getElementById('themeBtn').onclick = () => this.renderer.toggleTheme();
        document.querySelector('.server-list-box button').onclick = () => this.network.fetchServerList();

        // 開始遊戲迴圈
        this.startInputLoop();
        this.startRenderLoop();
        this.startFpsCounter();

        // 載入伺服器列表
        this.network.fetchServerList();
    }

    // --- 網路事件處理 ---

    onConnected() {
        this.isPlaying = false;
        this.isSpectating = false;
        this.isDead = true;
        this.localPlayers = {};
        this.showMenu("Play Game");
    }

    onDisconnected() {
        this.isPlaying = false;
        this.isSpectating = false;
        this.isDead = true;
        this.localPlayers = {};
        this.showMenu("Play Game");
    }

    onMessage(msg) {
        switch (msg.type) {
            case 'init':
                this.myId = msg.id;
                this.mapSize = msg.map;
                console.log("Map Initialized:", this.mapSize);
                break;

            case 'map_update':
                this.mapSize = msg.map;
                console.log("Map Resized:", this.mapSize);
                break;

            case 'update':
                this.processUpdate(msg.data);
                break;

            case 'death':
                this.isDead = true;
                this.isPlaying = false;
                this.isSpectating = false;
                this.showMenu("Respawn");
                break;
        }
    }

    processUpdate(data) {
        this.state.update(data);
        this.updateLeaderboard();

        const activeIds = new Set();
        data.players.forEach(p => {
            activeIds.add(p.id);

            if (!this.localPlayers[p.id]) {
                this.localPlayers[p.id] = {
                    ...p,
                    renderCells: p.cells.map(c => ({ ...c }))
                };
            } else {
                this.localPlayers[p.id].targetCells = p.cells;
                this.localPlayers[p.id].name = p.name;

                if (this.localPlayers[p.id].renderCells.length !== p.cells.length) {
                    this.localPlayers[p.id].renderCells = p.cells.map(c => ({ ...c }));
                }
            }
        });

        // 移除不存在的玩家
        Object.keys(this.localPlayers).forEach(id => {
            if (!activeIds.has(parseInt(id))) {
                delete this.localPlayers[id];
            }
        });
    }

    // --- UI 控制 ---

    showMenu(playText) {
        document.getElementById('menuOverlay').style.display = 'flex';
        document.getElementById('playBtn').innerText = playText;
    }

    hideMenu() {
        document.getElementById('menuOverlay').style.display = 'none';
    }

    toggleMenu() {
        const menu = document.getElementById('menuOverlay');
        if (menu.style.display === 'none') {
            this.showMenu(this.isPlaying && !this.isDead ? "Resume Game" : "Play Game");
        } else {
            if (this.isPlaying && !this.isDead) {
                this.hideMenu();
            }
        }
    }

    play() {
        if (this.isPlaying && !this.isDead) {
            this.hideMenu();
            return;
        }

        const name = document.getElementById('nickname').value || "Guest";
        this.network.join(name);
        this.isPlaying = true;
        this.isSpectating = false;
        this.isDead = false;
        this.hideMenu();
    }

    spectate() {
        this.network.spectate();
        this.isPlaying = false;
        this.isSpectating = true;
        this.specFollowMode = true;
        this.hideMenu();
    }

    updateLeaderboard() {
        const list = document.getElementById('lbList');
        list.innerHTML = '';

        this.state.leaderboard.forEach((p, i) => {
            const li = document.createElement('li');
            if (p.id === this.myId) li.className = 'lb-me';
            li.innerHTML = `<span>${i + 1}. ${p.name}</span> <span>${p.mass}</span>`;
            list.appendChild(li);
        });
    }

    // --- 遊戲迴圈 ---

    startInputLoop() {
        setInterval(() => {
            let tx, ty;
            const mouse = this.input.getMousePosition();

            if (this.isPlaying) {
                tx = isNaN(mouse.x) ? this.mapSize.w / 2 : mouse.x;
                ty = isNaN(mouse.y) ? this.mapSize.h / 2 : mouse.y;
            } else if (this.isSpectating) {
                tx = this.camX;
                ty = this.camY;
            }

            if (tx !== undefined && ty !== undefined) {
                this.network.sendInput(tx, ty);
            }
        }, CONFIG.INPUT_RATE);
    }

    startRenderLoop() {
        const draw = () => {
            this.update();
            this.render();
            this.frameCount++;
            requestAnimationFrame(draw);
        };
        requestAnimationFrame(draw);
    }

    startFpsCounter() {
        setInterval(() => {
            document.getElementById('fps').innerText = this.frameCount;
            this.frameCount = 0;
        }, 1000);
    }

    update() {
        let myTotalMass = 0;
        let center = { x: 0, y: 0, count: 0 };

        if (this.isPlaying && this.localPlayers[this.myId]) {
            const p = this.localPlayers[this.myId];
            p.renderCells.forEach(c => {
                center.x += c.x;
                center.y += c.y;
                center.count++;
                myTotalMass += c.m;
            });
        } else if (this.isSpectating) {
            if (this.specFollowMode && this.state.leaderboard.length > 0) {
                const topPlayer = this.state.leaderboard[0];
                if (this.localPlayers[topPlayer.id]) {
                    const p = this.localPlayers[topPlayer.id];
                    p.renderCells.forEach(c => {
                        center.x += c.x;
                        center.y += c.y;
                        center.count++;
                    });
                } else if (topPlayer.x !== undefined) {
                    center.x = topPlayer.x;
                    center.y = topPlayer.y;
                    center.count = 1;
                } else {
                    center.x = this.camX;
                    center.y = this.camY;
                    center.count = 1;
                }
            } else {
                const mouse = this.input.getMousePosition();
                center.x = this.camX + (mouse.x - this.camX) * 0.1;
                center.y = this.camY + (mouse.y - this.camY) * 0.1;
                center.count = 1;
            }
        }

        // 更新攝影機
        if (center.count > 0) {
            const tx = center.x / center.count;
            const ty = center.y / center.count;
            this.camX += (tx - this.camX) * CONFIG.CAMERA_SMOOTHING;
            this.camY += (ty - this.camY) * CONFIG.CAMERA_SMOOTHING;
        }

        if (this.isSpectating && !this.specFollowMode) {
            this.camX = Math.max(0, Math.min(this.mapSize.w, this.camX));
            this.camY = Math.max(0, Math.min(this.mapSize.h, this.camY));
        }

        // 更新縮放
        if (myTotalMass > 0) {
            const autoScale = 1.0 / Math.pow(Math.min(64.0 / this.renderer.massToRadius(myTotalMass), 1), 0.4);
            this.targetCamScale = (0.8 / autoScale) * this.input.userZoomModifier;
        } else {
            this.targetCamScale = 0.5 * this.input.userZoomModifier;
        }
        this.camScale += (this.targetCamScale - this.camScale) * 0.1;

        // 更新 HUD
        let specText = "";
        if (this.isSpectating) {
            specText = this.specFollowMode
                ? "(Spectating: Top Player - Q to Free)"
                : "(Spectating: Free Roam - Q to Follow)";
        }
        document.getElementById('massDisplay').innerText = this.isPlaying ? Math.floor(myTotalMass) : specText;
    }

    render() {
        this.renderer.clear();
        this.renderer.beginCamera(this.camX, this.camY, this.camScale);

        // 繪製背景
        this.renderer.drawBackground(this.mapSize);

        // 繪製食物
        this.renderer.drawFood(this.state.food);

        // 繪製拋射物
        this.renderer.drawEjected(this.state.ejected);

        // 繪製病毒
        this.renderer.drawViruses(this.state.viruses);

        // 繪製細胞（插值）
        const sortedCells = [];
        for (let pid in this.localPlayers) {
            const p = this.localPlayers[pid];
            if (p.targetCells) {
                for (let i = 0; i < p.renderCells.length; i++) {
                    const rc = p.renderCells[i];
                    const tc = p.targetCells[i] || rc;

                    // 插值
                    rc.x += (tc.x - rc.x) * CONFIG.INTERPOLATION_FACTOR;
                    rc.y += (tc.y - rc.y) * CONFIG.INTERPOLATION_FACTOR;
                    rc.m += (tc.m - rc.m) * CONFIG.INTERPOLATION_FACTOR;

                    sortedCells.push({
                        ...rc,
                        name: p.name,
                        isMine: parseInt(pid) === this.myId
                    });
                }
            }
        }

        // 按質量排序（小的先繪製）
        sortedCells.sort((a, b) => a.m - b.m);
        this.renderer.drawCells(sortedCells, this.myId);

        this.renderer.endCamera();
    }
}

// 啟動遊戲
window.addEventListener('DOMContentLoaded', () => {
    window.game = new Game();
});

export default Game;
