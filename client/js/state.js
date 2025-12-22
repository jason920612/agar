/**
 * 遊戲狀態管理
 */
class GameState {
    constructor() {
        this.reset();
    }

    reset() {
        this.players = [];
        this.food = [];
        this.viruses = [];
        this.ejected = [];
        this.leaderboard = [];
    }

    update(data) {
        this.food = data.food;
        this.viruses = data.viruses;
        this.ejected = data.ejected;
        this.leaderboard = data.leaderboard;
    }
}

export default GameState;
