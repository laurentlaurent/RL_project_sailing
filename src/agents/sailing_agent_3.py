import numpy as np
from agents.base_agent import BaseAgent

class OptimizedSailingAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.grid_size = (32, 32)
        self.goal = np.array([16, 31])
        self.strategies = ['direct', 'tacking', 'beam']
        self.reset()

    def reset(self):
        self.pos_history = []
        self.wind_data = []
        self.best_tack = None

    def act(self, obs):
        pos = np.array([obs[0], obs[1]])
        vel = np.array([obs[2], obs[3]])
        wind = np.array([obs[4], obs[5]])
        wind_field = obs[6:].reshape(*self.grid_size, 2)
        
        self.pos_history.append(pos.copy())
        self.wind_data.append(wind.copy())
        
        if self._near_goal(pos): return 8
        
        wind_analysis = self._analyze_wind(wind_field)
        action = self._select_strategy(pos, vel, wind, wind_analysis)
        
        return action

    def _near_goal(self, pos):
        return np.linalg.norm(pos - self.goal) < 2

    def _analyze_wind(self, wind_field):
        strength = np.linalg.norm(wind_field, axis=2)
        return {
            'mean': np.mean(strength),
            'max': np.max(strength),
            'direction': np.mean(wind_field, axis=(0,1))
        }

    def _select_strategy(self, pos, vel, wind, wind_info):
        if self._is_stuck(pos):
            return self._escape_move(pos, wind)
            
        goal_dir = self.goal - pos
        wind_dir = -wind / (np.linalg.norm(wind) + 1e-6)
        
        if np.dot(goal_dir/np.linalg.norm(goal_dir), wind_dir) < 0.7:
            return self._tacking_strategy(pos, wind_dir)
        
        return self._optimized_move(pos, wind, goal_dir)

    def _is_stuck(self, pos):
        if len(self.pos_history) < 5: return False
        return np.std(self.pos_history[-5:], axis=0).mean() < 0.5

    def _escape_move(self, pos, wind):
        directions = [np.array([dx, dy]) for dx, dy in [
            (0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1)
        ]]
        scores = [self._move_score(pos+d, wind) for d in directions]
        return np.argmax(scores)

    def _move_score(self, new_pos, wind):
        if any(new_pos < 0) or any(new_pos >= self.grid_size): return -100
        wind_eff = self._wind_efficiency(new_pos - self.goal, wind)
        goal_dist = -np.linalg.norm(new_pos - self.goal)
        return wind_eff * 0.7 + goal_dist * 0.3

    def _tacking_strategy(self, pos, wind_dir):
        angles = np.linspace(np.pi/4, 3*np.pi/4, 8)
        directions = [wind_dir + np.array([np.cos(a), np.sin(a)]) for a in angles]
        scores = [self._tack_score(pos, d) for d in directions]
        return np.argmax(scores)

    def _tack_score(self, pos, direction):
        proj = np.dot(direction, (self.goal - pos))
        wind_eff = self._wind_efficiency(direction, -direction)
        return proj * 0.6 + wind_eff * 0.4

    def _optimized_move(self, pos, wind, goal_dir):
        actions = []
        for action in range(8):
            move = self._action_vector(action)
            eff = self._wind_efficiency(move, wind)
            progress = np.dot(move, goal_dir)
            actions.append((action, eff * 0.8 + progress * 0.2))
        return max(actions, key=lambda x: x[1])[0]

    def _wind_efficiency(self, direction, wind):
        dir_norm = direction / (np.linalg.norm(direction) + 1e-6)
        wind_norm = wind / (np.linalg.norm(wind) + 1e-6)
        angle = np.arccos(np.clip(np.dot(dir_norm, wind_norm), -1, 1))
        return 1.0 - min(angle/np.pi, 0.5)*2

    def _action_vector(self, action):
        return np.array([
            (0,1), (1,1), (1,0), (1,-1),
            (0,-1), (-1,-1), (-1,0), (-1,1)
        ][action])