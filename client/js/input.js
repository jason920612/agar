/**
 * 輸入處理器
 */
class InputHandler {
    constructor(game) {
        this.game = game;
        this.mouseX = 0;
        this.mouseY = 0;
        this.userZoomModifier = 1.0;

        this.bindEvents();
    }

    bindEvents() {
        const canvas = document.getElementById('gameCanvas');

        // 滑鼠移動
        canvas.addEventListener('mousemove', (e) => {
            const rect = canvas.getBoundingClientRect();
            const game = this.game;

            this.mouseX = (e.clientX - rect.left - canvas.width / 2) / game.camScale + game.camX;
            this.mouseY = (e.clientY - rect.top - canvas.height / 2) / game.camScale + game.camY;
        });

        // 滾輪縮放
        window.addEventListener('wheel', (e) => {
            if (this.game.isPlaying || this.game.isSpectating) {
                this.userZoomModifier += e.deltaY * -0.001;
                this.userZoomModifier = Math.max(0.3, Math.min(2.0, this.userZoomModifier));
            }
        }, { passive: true });

        // 鍵盤輸入
        window.addEventListener('keydown', (e) => {
            if (e.code === 'Escape') {
                this.game.toggleMenu();
            }

            if (e.code === 'KeyQ' && this.game.isSpectating) {
                this.game.specFollowMode = !this.game.specFollowMode;
            }

            if (!this.game.isPlaying || this.game.isDead) return;

            if (e.code === 'Space') {
                this.game.network.split();
            }
            if (e.code === 'KeyW') {
                this.game.network.eject();
            }
        });
    }

    getMousePosition() {
        return { x: this.mouseX, y: this.mouseY };
    }
}

export default InputHandler;
