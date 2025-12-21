/**
 * 遊戲渲染器
 */
class Renderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.isDarkTheme = true;
    }

    resize() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }

    massToRadius(mass) {
        return 6 * Math.sqrt(mass);
    }

    clear() {
        this.resize();
    }

    beginCamera(camX, camY, camScale) {
        this.ctx.save();
        this.ctx.translate(this.canvas.width / 2, this.canvas.height / 2);
        this.ctx.scale(camScale, camScale);
        this.ctx.translate(-camX, -camY);
    }

    endCamera() {
        this.ctx.restore();
    }

    drawBackground(mapSize) {
        const bgColor = this.isDarkTheme ? '#111' : '#f2fbff';
        const gridColor = this.isDarkTheme ? '#333' : '#ccc';
        const borderColor = this.isDarkTheme ? '#555' : '#333';

        this.ctx.fillStyle = bgColor;
        this.ctx.fillRect(0, 0, mapSize.w, mapSize.h);

        // 網格
        this.ctx.strokeStyle = gridColor;
        this.ctx.lineWidth = 1;
        this.ctx.beginPath();
        for (let x = 0; x <= mapSize.w; x += 50) {
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, mapSize.h);
        }
        for (let y = 0; y <= mapSize.h; y += 50) {
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(mapSize.w, y);
        }
        this.ctx.stroke();

        // 邊框
        this.ctx.strokeStyle = borderColor;
        this.ctx.lineWidth = 5;
        this.ctx.strokeRect(0, 0, mapSize.w, mapSize.h);
    }

    drawFood(food) {
        food.forEach(f => {
            this.ctx.fillStyle = f.color;
            this.ctx.beginPath();
            this.ctx.arc(f.x, f.y, this.massToRadius(f.mass), 0, Math.PI * 2);
            this.ctx.fill();
        });
    }

    drawEjected(ejected) {
        const strokeColor = this.isDarkTheme ? '#eee' : '#333';

        ejected.forEach(e => {
            this.ctx.fillStyle = e.c;
            this.ctx.strokeStyle = strokeColor;
            this.ctx.lineWidth = 2;
            this.ctx.beginPath();
            this.ctx.arc(e.x, e.y, this.massToRadius(15), 0, Math.PI * 2);
            this.ctx.fill();
            this.ctx.stroke();
        });
    }

    drawViruses(viruses) {
        viruses.forEach(v => {
            this.ctx.fillStyle = '#33ff33';
            this.ctx.strokeStyle = '#22aa22';
            this.ctx.lineWidth = 4;
            this.ctx.beginPath();

            const r = this.massToRadius(v.m);
            for (let i = 0; i < 20; i++) {
                const a = (i / 20) * Math.PI * 2;
                const rad = i % 2 === 0 ? r : r * 0.92;
                this.ctx.lineTo(v.x + Math.cos(a) * rad, v.y + Math.sin(a) * rad);
            }
            this.ctx.closePath();
            this.ctx.fill();
            this.ctx.stroke();
        });
    }

    drawCells(sortedCells, myId) {
        sortedCells.forEach(c => {
            const r = this.massToRadius(c.m);
            this.ctx.fillStyle = c.c;
            this.ctx.strokeStyle = c.isMine ? '#fff' : (this.isDarkTheme ? '#000' : '#333');
            this.ctx.lineWidth = c.isMine ? 4 : 2;

            this.ctx.beginPath();
            this.ctx.arc(c.x, c.y, r, 0, Math.PI * 2);
            this.ctx.fill();
            this.ctx.stroke();

            // 名稱
            if (r > 15) {
                this.ctx.fillStyle = '#fff';
                this.ctx.strokeStyle = '#000';
                this.ctx.lineWidth = 2;
                this.ctx.font = `bold ${Math.max(12, r * 0.4)}px sans-serif`;
                this.ctx.textAlign = 'center';
                this.ctx.textBaseline = 'middle';
                this.ctx.strokeText(c.name, c.x, c.y);
                this.ctx.fillText(c.name, c.x, c.y);
            }
        });
    }

    toggleTheme() {
        this.isDarkTheme = !this.isDarkTheme;
        document.body.style.backgroundColor = this.isDarkTheme ? '#111' : '#fff';
    }
}

export default Renderer;
