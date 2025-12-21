/**
 * 網路連線管理
 */
import CONFIG from './config.js';

class NetworkManager {
    constructor(game) {
        this.game = game;
        this.socket = null;
        this.currentServerUrl = null;
        this.servers = [];
        this.reconnectTimeout = null;
    }

    async fetchServerList() {
        const container = document.getElementById('serverListContainer');
        container.innerHTML = '<div style="padding:10px; color:#888;">Fetching...</div>';

        try {
            const res = await fetch(CONFIG.MASTER_URL);
            this.servers = await res.json();
            this.renderServerList();

            if (!this.currentServerUrl && this.servers.length > 0) {
                this.currentServerUrl = this.servers[0].url;
                this.connect();
            }
        } catch (e) {
            container.innerHTML = '<div style="padding:10px; color:#f00;">Master Offline</div>';
        }
    }

    renderServerList() {
        const container = document.getElementById('serverListContainer');
        container.innerHTML = '';

        if (this.servers.length === 0) {
            container.innerHTML = '<div style="padding:10px;">No servers.</div>';
            return;
        }

        this.servers.forEach((s) => {
            const div = document.createElement('div');
            div.className = 'server-item' + (s.url === this.currentServerUrl ? ' selected' : '');
            div.innerHTML = `
                <span class="status-dot ${s.status === 'online' ? 'status-online' : ''}"></span>
                <span>${s.name}</span>
                <small>${s.players || 0}/${s.max_players}</small>
            `;

            if (s.status === 'online') {
                div.onclick = () => {
                    this.currentServerUrl = s.url;
                    this.connect();
                    this.renderServerList();
                };
            }

            container.appendChild(div);
        });
    }

    connect() {
        if (this.socket) {
            this.socket.onclose = null;
            this.socket.close();
        }

        if (!this.currentServerUrl) return;

        this.socket = new WebSocket(this.currentServerUrl);

        this.socket.onopen = () => {
            console.log("Connected to " + this.currentServerUrl);
            this.game.onConnected();
        };

        this.socket.onmessage = (e) => {
            const msg = JSON.parse(e.data);
            this.game.onMessage(msg);
        };

        this.socket.onclose = () => {
            console.log("Disconnected - Reconnecting...");
            this.game.onDisconnected();

            if (this.reconnectTimeout) clearTimeout(this.reconnectTimeout);
            this.reconnectTimeout = setTimeout(() => this.connect(), 2000);
        };
    }

    send(data) {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            this.socket.send(JSON.stringify(data));
        }
    }

    join(name) {
        this.send({ type: 'join', name: name });
    }

    spectate() {
        this.send({ type: 'spectate' });
    }

    sendInput(x, y) {
        this.send({ type: 'input', x: x, y: y });
    }

    split() {
        this.send({ type: 'split' });
    }

    eject() {
        this.send({ type: 'eject' });
    }
}

export default NetworkManager;
