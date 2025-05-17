"""
Sailing Challenge - Agent de Navigation Avancé
Un agent hybride combinant analyse du champ de vent, planification de trajectoire,
et techniques de navigation optimisées pour le Sailing Challenge.
"""

import numpy as np
import heapq
from typing import Tuple, List, Dict, Set, Optional, Any
from agents.base_agent import BaseAgent

class SailingAgent(BaseAgent):
    """
    Agent hybride de navigation à voile qui utilise des stratégies avancées
    pour naviguer efficacement dans des conditions de vent variables.
    """
    
    def __init__(self):
        """Initialisation de l'agent avec les paramètres nécessaires."""
        super().__init__()
        
        # Paramètres principaux
        self.grid_size = (32, 32)  # Taille par défaut de la grille
        self.goal_position = np.array([self.grid_size[0] // 2, self.grid_size[1] - 1])
        
        # Paramètres de navigation
        self.min_wind_angle = np.radians(45)  # Angle minimum par rapport au vent (zone interdite)
        self.optimal_wind_angle = np.radians(90)  # Angle optimal par rapport au vent
        self.tacking_threshold = np.radians(30)  # Seuil pour décider de virer de bord        
        
        # Structure du champ de vent et de l'état
        self.wind_field = None
        self.wind_analysis = None  # Stockage de l'analyse du champ de vent
        self.previous_positions = []  # Pour détecter les blocages
        self.blocked_positions = set()  # Positions à éviter (blocages précédents)
        self.planned_path = []  # Chemin planifié
        self.current_strategy = None  # Stratégie actuelle
        self.tack_direction = 1  # Direction initiale de virement (1 ou -1)
        self.tack_count = 0  # Compteur de virements
        self.stuck_counter = 0  # Compteur pour détecter les situations de blocage
        
        # Paramètres pour les laylines
        self.laylines = {"port": None, "starboard": None}  # Points d'intersection avec la ligne d'arrivée
        self.best_tack = None  # Meilleur bord pour atteindre l'objectif
        
        # Cache pour les calculs coûteux
        self.vmg_cache = {}  # Cache pour les calculs de VMG
        self.efficiency_cache = {}  # Cache pour les calculs d'efficacité
        
        # Paramètres de performance
        self.velocity_memory = 0.7  # Facteur de mémoire pour la vitesse (inertie)
        self.last_action = None
        self.last_vmg = 0  # Dernière valeur de VMG pour comparaison
        self.np_random = np.random.default_rng()
        
        # Paramètres d'apprentissage par renforcement
        self.q_values = {}  # (wind_bin, position_bin, velocity_bin) -> [valeurs d'action]
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.exploration_rate = 0.1  # Taux d'exploration (epsilon)
        self.state_bins = 8  # Nombre de bins pour la discrétisation
        
        self.reset()
        
    def reset(self) -> None:
        """Réinitialisation de l'agent au début d'un nouvel épisode."""
        self.previous_positions = []
        self.blocked_positions = set()
        self.planned_path = []
        self.current_strategy = None
        self.tack_direction = 1
        self.tack_count = 0
        self.stuck_counter = 0
        self.laylines = {"port": None, "starboard": None}
        self.best_tack = None
        self.vmg_cache = {}
        self.efficiency_cache = {}
        self.last_action = None
        self.last_vmg = 0
        self.wind_analysis = None
        
    def seed(self, seed: Optional[int] = None) -> None:
        """Configuration de la graine aléatoire pour la reproductibilité."""
        self.np_random = np.random.default_rng(seed)
        
    def act(self, observation: np.ndarray) -> int:
        """
        Sélectionne l'action optimale basée sur l'observation actuelle.
        
        Args:
            observation: Tableau numpy contenant l'observation actuelle.
                Format: [x, y, vx, vy, wx, wy, wind_field_flattened]
                
        Returns:
            action: Un entier dans [0, 8] représentant l'action à prendre
        """
        # Extraction des données d'observation
        position = np.array([observation[0], observation[1]])
        velocity = np.array([observation[2], observation[3]])
        local_wind = np.array([observation[4], observation[5]])
        
        # Extraction et reconstitution du champ de vent
        wind_field_flat = observation[6:]
        self.grid_size = self._infer_grid_size(len(wind_field_flat))
        self.goal_position = np.array([self.grid_size[0] // 2, self.grid_size[1] - 1])
        self.wind_field = self._reshape_wind_field(wind_field_flat)
        
        # Analyser le champ de vent pour obtenir des informations sur la structure
        self.wind_analysis = self._analyze_wind_patterns()
        
        # Calculer les laylines (lignes de remontée optimales vers l'objectif)
        self._calculate_laylines(position, local_wind)
        
        # Mise à jour de l'historique des positions pour la détection de blocage
        self._update_position_history(position)
        
        # Vérification si nous sommes proches du but
        if np.linalg.norm(position - self.goal_position) < 2:
            # Direction directe vers l'objectif pour finir
            return self._get_action_towards_goal(position, velocity, local_wind)
        
        # Détection et gestion des situations de blocage
        if self._detect_stuck(position):
            # Sortie d'une situation de blocage
            return self._handle_stuck_situation(position, velocity, local_wind)
        
        # Vérifier si nous sommes sur une layline et pouvons aller directement à l'objectif
        if self._on_layline(position):
            return self._follow_layline(position, velocity, local_wind)
        
        # Méta-décision: combiner plusieurs stratégies et choisir la meilleure
        return self._meta_decision(position, velocity, local_wind)
    
    def _infer_grid_size(self, wind_field_length: int) -> Tuple[int, int]:
        """Déduire la taille de la grille à partir de la longueur du champ de vent aplati."""
        # Le champ de vent contient 2 composantes (x,y) par cellule
        total_cells = wind_field_length // 2
        
        # Supposer une grille carrée par défaut, mais essayer d'inférer les dimensions réelles
        grid_dim = int(np.sqrt(total_cells))
        return (grid_dim, grid_dim)
    
    def _reshape_wind_field(self, wind_field_flat: np.ndarray) -> np.ndarray:
        """Reconstitue le champ de vent 2D à partir de sa version aplatie."""
        # Nombre total de cellules
        height, width = self.grid_size
        total_cells = height * width
        
        # Vérifier que la taille du champ de vent correspond à nos attentes
        if len(wind_field_flat) != total_cells * 2:
            # Ajuster la taille de la grille si nécessaire
            self.grid_size = self._infer_grid_size(len(wind_field_flat))
            height, width = self.grid_size
            total_cells = height * width
            
        # Restructurer le champ de vent en une grille 3D: [y, x, (wx, wy)]
        return wind_field_flat.reshape(height, width, 2)
    
    def _analyze_wind_patterns(self) -> Dict[str, Any]:
        """
        Analyse le champ de vent pour identifier les structures favorables et défavorables.
        Utilisée pour améliorer la planification de trajectoire.
        
        Returns:
            Dict contenant les informations sur les structures de vent:
            - 'favorable_regions': coordonnées des zones de vent fort favorable
            - 'unfavorable_regions': coordonnées des zones de vent défavorable
            - 'wind_gradients': informations sur les gradients de vent
            - 'avg_wind_strength': force moyenne du vent
        """
        # Calculer les gradients de vent (magnitude et direction)
        gradient_x = np.gradient(self.wind_field[:,:,0])
        gradient_y = np.gradient(self.wind_field[:,:,1])
        
        # Calculer la magnitude du vent en chaque point
        wind_strength = np.linalg.norm(self.wind_field, axis=2)
        avg_strength = np.mean(wind_strength)
        
        # Identifier les corridors de vent favorable (vent plus fort que la moyenne)
        favorable_mask = wind_strength > avg_strength * 1.2
        favorable_regions = list(zip(*np.where(favorable_mask)))
        
        # Identifier les zones de vent faible à éviter
        unfavorable_mask = wind_strength < avg_strength * 0.8
        unfavorable_regions = list(zip(*np.where(unfavorable_mask)))
        
        # Calculer les variations de direction du vent
        # (utile pour repérer les zones de transition)
        wind_directions = np.arctan2(self.wind_field[:,:,1], self.wind_field[:,:,0])
        direction_gradients = np.gradient(wind_directions)
        direction_changes = np.sqrt(direction_gradients[0]**2 + direction_gradients[1]**2)
        
        # Identifier les zones avec changements significatifs de direction
        transition_regions = list(zip(*np.where(direction_changes > np.percentile(direction_changes, 80))))
        
        return {
            'favorable_regions': favorable_regions,
            'unfavorable_regions': unfavorable_regions,
            'transition_regions': transition_regions,
            'wind_gradients': (gradient_x, gradient_y),
            'direction_changes': direction_changes,
            'avg_wind_strength': avg_strength
        }
    
    def _calculate_laylines(self, position: np.ndarray, local_wind: np.ndarray) -> None:
        """
        Calcule les laylines pour atteindre l'objectif en remontant au vent.
        Les laylines sont les lignes droites qui, suivies avec un angle optimal par rapport au vent,
        mènent directement à l'objectif.
        """
        # Si le vent est quasi-nul, pas besoin de laylines
        if np.linalg.norm(local_wind) < 0.1:
            self.laylines = {"port": None, "starboard": None}
            self.best_tack = None
            return
        
        # Direction d'où vient le vent
        wind_from = -local_wind / np.linalg.norm(local_wind)
        
        # Angle optimal pour remonter au vent (typiquement 45°)
        tack_angle = np.radians(45)
        
        # Calculer les deux directions de tacking (bâbord et tribord)
        port_tack_dir = self._rotate_vector(wind_from, tack_angle)
        starboard_tack_dir = self._rotate_vector(wind_from, -tack_angle)
        
        # Calculer les intersections avec la ligne d'objectif
        x_goal = self.goal_position[0]
        y_goal = self.goal_position[1]
        
        # Point sur la ligne d'objectif pour bâbord
        if abs(port_tack_dir[0]) > 1e-6:  # Éviter division par zéro
            t_port = (x_goal - position[0]) / port_tack_dir[0]
            y_port_intersect = position[1] + t_port * port_tack_dir[1]
            
            # Vérifier si l'intersection est valide (sur la ligne d'objectif)
            if 0 <= y_port_intersect <= y_goal:
                port_intersect = np.array([x_goal, y_port_intersect])
            else:
                # Pas d'intersection valide
                port_intersect = None
        else:
            port_intersect = None
        
        # Point sur la ligne d'objectif pour tribord
        if abs(starboard_tack_dir[0]) > 1e-6:  # Éviter division par zéro
            t_starboard = (x_goal - position[0]) / starboard_tack_dir[0]
            y_starboard_intersect = position[1] + t_starboard * starboard_tack_dir[1]
            
            # Vérifier si l'intersection est valide (sur la ligne d'objectif)
            if 0 <= y_starboard_intersect <= y_goal:
                starboard_intersect = np.array([x_goal, y_starboard_intersect])
            else:
                # Pas d'intersection valide
                starboard_intersect = None
        else:
            starboard_intersect = None
        
        # Stocker les laylines
        self.laylines = {"port": port_intersect, "starboard": starboard_intersect}
        
        # Déterminer le meilleur bord pour atteindre l'objectif
        if port_intersect is not None and starboard_intersect is not None:
            # Choisir le bord qui minimise la distance totale
            port_distance = np.linalg.norm(port_intersect - position) + np.linalg.norm(self.goal_position - port_intersect)
            starboard_distance = np.linalg.norm(starboard_intersect - position) + np.linalg.norm(self.goal_position - starboard_intersect)
            
            if port_distance < starboard_distance:
                self.best_tack = "port"
            else:
                self.best_tack = "starboard"
        elif port_intersect is not None:
            self.best_tack = "port"
        elif starboard_intersect is not None:
            self.best_tack = "starboard"
        else:
            self.best_tack = None
    
    def _on_layline(self, position: np.ndarray) -> bool:
        """
        Vérifie si nous sommes actuellement sur une layline valide.
        """
        if self.best_tack is None or self.laylines[self.best_tack] is None:
            return False
        
        # Calculer la distance à la layline
        layline_point = self.laylines[self.best_tack]
        
        # Vecteur de la layline
        layline_vector = layline_point - position
        layline_distance = np.linalg.norm(layline_vector)
        
        # Si nous sommes très proches de la layline, considérer que nous sommes dessus
        return layline_distance < 3.0
    
    def _follow_layline(self, position: np.ndarray, velocity: np.ndarray, local_wind: np.ndarray) -> int:
        """
        Suit une layline pour atteindre l'objectif en remontant au vent.
        """
        if self.best_tack is None or self.laylines[self.best_tack] is None:
            # Pas de layline valide, utiliser une autre stratégie
            return self._tacking_strategy(position, velocity, local_wind)
        
        layline_point = self.laylines[self.best_tack]
        
        # Direction vers le point de la layline
        direction_to_layline = layline_point - position
        
        # Si nous sommes proches du point de layline, viser directement l'objectif
        if np.linalg.norm(direction_to_layline) < 1.0:
            direction_to_goal = self.goal_position - position
            return self._find_best_action_vmg(position, local_wind, direction_to_goal)
        
        # Sinon, suivre la direction de la layline
        return self._find_best_action_vmg(position, local_wind, direction_to_layline)
    
    def _update_position_history(self, position: np.ndarray) -> None:
        """Met à jour l'historique des positions pour détecter les blocages."""
        position_tuple = tuple(position.astype(int))
        
        # Ajouter la position actuelle à l'historique
        self.previous_positions.append(position_tuple)
        
        # Limiter la taille de l'historique
        if len(self.previous_positions) > 20:
            self.previous_positions.pop(0)
    
    def _detect_stuck(self, position: np.ndarray) -> bool:
        """Détecte si l'agent est bloqué en analysant l'historique des positions."""
        if len(self.previous_positions) < 10:
            return False
        
        # Convertir la position actuelle en tuple pour la comparaison
        position_tuple = tuple(position.astype(int))
        
        # Vérifier s'il y a trop de répétitions de la même position
        position_counts = {}
        for pos in self.previous_positions[-10:]:
            position_counts[pos] = position_counts.get(pos, 0) + 1
        
        # Si une position apparaît plus de 3 fois dans les 10 dernières positions, on est bloqué
        for pos, count in position_counts.items():
            if count >= 4:
                self.stuck_counter += 1
                self.blocked_positions.add(pos)
                return True
        
        # Si on a bougé normalement, réduire le compteur de blocage
        self.stuck_counter = max(0, self.stuck_counter - 1)
        return False
    
    def _handle_stuck_situation(self, position: np.ndarray, velocity: np.ndarray, local_wind: np.ndarray) -> int:
        """
        Stratégie pour sortir d'une situation de blocage.
        Utilise une combinaison de mouvements aléatoires et dirigés pour échapper au blocage.
        """
        # Réinitialiser le chemin planifié
        self.planned_path = []
        
        if self.stuck_counter > 3:
            # En cas de blocage persistant, changer radicalement de direction
            actions = list(range(8))  # Toutes les directions sauf "rester sur place"
            self.np_random.shuffle(actions)
            # Sélectionner une action différente de la dernière
            for action in actions:
                if action != self.last_action:
                    self.last_action = action
                    return action
        
        # Trouver la direction qui nous éloigne le plus des positions bloquées
        best_action = None
        max_distance = -1
        
        for action in range(8):  # Toutes les directions sauf "rester sur place"
            direction = self._action_to_direction(action)
            new_position = position + direction
            
            # Calculer la distance moyenne aux positions bloquées
            total_distance = 0
            for blocked_pos in self.blocked_positions:
                blocked_pos_array = np.array(blocked_pos)
                dist = np.linalg.norm(new_position - blocked_pos_array)
                total_distance += dist
            
            avg_distance = total_distance / max(1, len(self.blocked_positions))
            
            # Préférer les mouvements qui nous rapprochent aussi de l'objectif
            goal_factor = 1 - np.linalg.norm(new_position - self.goal_position) / np.linalg.norm(self.grid_size)
            combined_score = avg_distance * (1 + 0.5 * goal_factor)
            
            if combined_score > max_distance:
                max_distance = combined_score
                best_action = action
        
        # Si aucune action n'est trouvée, essayer une approche différente
        if best_action is None:
            return self._exploration_strategy(position, velocity, local_wind)
        
        self.last_action = best_action
        return best_action
    
    def _meta_decision(self, position: np.ndarray, velocity: np.ndarray, local_wind: np.ndarray) -> int:
        """
        Méta-décision qui évalue plusieurs stratégies et sélectionne la meilleure action.
        Combine les avantages de différentes approches.
        """
        actions_with_scores = []
        
        # 1. Sélectionner la stratégie optimale en fonction des conditions
        strategy = self._select_optimal_strategy(position, velocity, local_wind)
        self.current_strategy = strategy
        
        # 2. Évaluer l'action de la stratégie primaire
        primary_action = self._get_strategy_action(strategy, position, velocity, local_wind)
        primary_score = self._evaluate_action(primary_action, position, velocity, local_wind)
        actions_with_scores.append((primary_action, primary_score, strategy))
        
        # 3. Évaluer d'autres stratégies alternatives
        alternative_strategies = [s for s in ["direct", "tacking", "beam_reach", "planned_path"] if s != strategy]
        for alt_strategy in alternative_strategies[:2]:  # Limiter à 2 alternatives pour l'efficacité
            alt_action = self._get_strategy_action(alt_strategy, position, velocity, local_wind)
            alt_score = self._evaluate_action(alt_action, position, velocity, local_wind) * 0.9  # Léger désavantage
            actions_with_scores.append((alt_action, alt_score, alt_strategy))
        
        # 4. Considérer un facteur d'exploration (epsilon-greedy)
        if self.np_random.random() < self.exploration_rate:
            random_action = self.np_random.integers(0, 8)
            explore_score = self._evaluate_action(random_action, position, velocity, local_wind) * 1.1  # Bonus d'exploration
            actions_with_scores.append((random_action, explore_score, "explore"))
        
        # 5. Sélectionner la meilleure action
        best_action, best_score, best_strategy = max(actions_with_scores, key=lambda x: x[1])
        
        # 6. Mettre à jour la stratégie actuelle
        self.current_strategy = best_strategy
        self.last_action = best_action
        
        return best_action
    
    def _get_strategy_action(self, strategy: str, position: np.ndarray, velocity: np.ndarray, local_wind: np.ndarray) -> int:
        """
        Obtient l'action recommandée par une stratégie spécifique.
        """
        if strategy == "direct":
            return self._direct_to_goal(position, velocity, local_wind)
        elif strategy == "tacking":
            return self._tacking_strategy(position, velocity, local_wind)
        elif strategy == "beam_reach":
            return self._beam_reach_strategy(position, velocity, local_wind)
        elif strategy == "planned_path":
            # Vérifier si nous avons un chemin planifié, sinon en créer un
            if not self.planned_path:
                self._plan_path(position)
            return self._follow_planned_path(position, velocity, local_wind)
        elif strategy == "explore":
            return self._exploration_strategy(position, velocity, local_wind)
        else:
            # Stratégie par défaut: mouvement direct si possible ou tacking si nécessaire
            return self._adaptive_movement(position, velocity, local_wind)
    
    def _evaluate_action(self, action: int, position: np.ndarray, velocity: np.ndarray, local_wind: np.ndarray) -> float:
        """
        Évalue la qualité d'une action en fonction de multiples facteurs.
        
        Returns:
            score: Valeur numérique indiquant la qualité de l'action
        """
        # Convertir l'action en direction
        direction = self._action_to_direction(action)
        new_position = position + direction
        
        # Vérifier si la nouvelle position est valide
        if (new_position[0] < 0 or new_position[0] >= self.grid_size[0] or
            new_position[1] < 0 or new_position[1] >= self.grid_size[1]):
            return -100.0  # Très mauvais score pour les positions hors limites
        
        # Vérifier si c'est une position bloquée connue
        if tuple(new_position.astype(int)) in self.blocked_positions:
            return -50.0
        
        # Direction vers l'objectif
        goal_direction = self.goal_position - position
        goal_distance = np.linalg.norm(goal_direction)
        if goal_distance > 0:
            goal_direction = goal_direction / goal_distance
        
        # 1. Progression vers l'objectif (VMG)
        vmg = np.dot(direction, goal_direction)
        
        # 2. Efficacité de navigation
        efficiency = self._calculate_sailing_efficiency(direction, local_wind)
        
        # 3. Facteur d'inertie (continuité du mouvement)
        inertia_factor = 0.0
        if self.last_action is not None and self.last_action < 8:
            last_direction = self._action_to_direction(self.last_action)
            direction_similarity = np.dot(direction, last_direction)
            inertia_factor = 0.2 * max(0, direction_similarity)
        
        # 4. Facteur de vent favorable
        wind_factor = 0.0
        new_pos_tuple = tuple(new_position.astype(int))
        if self.wind_analysis:
            if new_pos_tuple in self.wind_analysis.get('favorable_regions', set()):
                wind_factor = 0.3  # Bonus pour les zones de vent favorable
            elif new_pos_tuple in self.wind_analysis.get('unfavorable_regions', set()):
                wind_factor = -0.3  # Pénalité pour les zones de vent défavorable
        
        # 5. Proximité des laylines
        layline_factor = 0.0
        if self.best_tack and self.laylines[self.best_tack] is not None:
            layline_point = self.laylines[self.best_tack]
            layline_direction = layline_point - position
            layline_alignment = np.dot(direction, layline_direction) / (np.linalg.norm(layline_direction) + 1e-10)
            layline_factor = 0.2 * max(0, layline_alignment)
        
        # Combiner tous les facteurs
        score = (
            1.0 * vmg +  # Facteur principal: progression vers l'objectif
            0.8 * efficiency +  # Efficacité de navigation
            0.3 * inertia_factor +  # Continuité du mouvement
            wind_factor +  # Zones de vent favorable/défavorable
            layline_factor  # Alignement avec les laylines
        )
        
        return score
    
    def _select_optimal_strategy(self, position: np.ndarray, velocity: np.ndarray, local_wind: np.ndarray) -> str:
        """
        Sélectionne la stratégie optimale en fonction des conditions actuelles.
        
        Stratégies disponibles:
        - "direct": Mouvement direct vers l'objectif si possible
        - "tacking": Virement de bord pour remonter au vent
        - "beam_reach": Optimisation de l'angle par rapport au vent pour maximiser la vitesse
        - "planned_path": Suivre un chemin planifié dans le champ de vent
        - "explore": Exploration pour sortir des situations difficiles
        """
        # Direction vers l'objectif
        goal_direction = self.goal_position - position
        goal_distance = np.linalg.norm(goal_direction)
        
        # Si nous sommes très proches de l'objectif, aller directement vers celui-ci
        if goal_distance < 5:
            return "direct"
        
        # Calculer l'angle entre le vent et la direction de l'objectif
        if np.linalg.norm(local_wind) > 0 and np.linalg.norm(goal_direction) > 0:
            # Direction d'où vient le vent
            wind_from = -local_wind / np.linalg.norm(local_wind)
            goal_norm = goal_direction / np.linalg.norm(goal_direction)
            
            wind_goal_angle = np.arccos(np.clip(np.dot(wind_from, goal_norm), -1.0, 1.0))
            
            # Si l'objectif est près de la zone interdite du vent (près du vent)
            if wind_goal_angle < self.min_wind_angle * 1.2:
                # Si nous avons une layline valide, l'utiliser
                if self.best_tack and self.laylines[self.best_tack] is not None:
                    return "direct"  # Le meta-decision gérera l'utilisation des laylines
                
                # Sinon, besoin de louvoyage (tacking)
                return "tacking"
            
            # Si l'objectif est proche de l'angle optimal par rapport au vent
            elif abs(wind_goal_angle - self.optimal_wind_angle) < np.radians(20):
                # Utiliser la stratégie de navigation au près optimal (beam reach)
                return "beam_reach"
        
        # Si nous avons un chemin planifié et qu'il est encore valide
        if self.planned_path and len(self.planned_path) > 0:
            # Vérifier si le chemin est toujours valide (premier point pas trop loin)
            if np.linalg.norm(np.array(self.planned_path[0]) - position) < 5:
                return "planned_path"
        
        # Si nous sommes bloqués ou si nous n'avons pas progressé récemment
        if self.stuck_counter > 0:
            return "explore"
        
        # Si aucune condition spécifique n'est remplie, planifier un nouveau chemin
        self._plan_path(position)
        if self.planned_path and len(self.planned_path) > 0:
            return "planned_path"
        
        # Par défaut, utiliser une approche adaptative
        return "direct"
    
    def _direct_to_goal(self, position: np.ndarray, velocity: np.ndarray, local_wind: np.ndarray) -> int:
        """Stratégie de mouvement direct vers l'objectif."""
        goal_direction = self.goal_position - position
        
        # Calculer le meilleur mouvement en tenant compte du vent et de l'efficacité de navigation
        best_action = self._find_best_action_vmg(position, local_wind, goal_direction)
        self.last_action = best_action
        return best_action
    
    def _tacking_strategy(self, position: np.ndarray, velocity: np.ndarray, local_wind: np.ndarray) -> int:
        """
        Stratégie de louvoyage (tacking) pour remonter au vent.
        Cette stratégie permet de zigzaguer pour avancer contre le vent.
        """
        goal_direction = self.goal_position - position
        
        # Direction du vent inversée (d'où il vient)
        wind_from = -local_wind / (np.linalg.norm(local_wind) + 1e-10)
        
        # Déterminer l'angle optimal pour le louvoyage (environ 45° par rapport au vent)
        tack_angle = np.radians(45)  # Angle classique de remontée au vent
        
        # Calculer deux directions possibles de louvoyage (bâbord et tribord)
        # Rotation à gauche du vent
        left_tack_dir = self._rotate_vector(wind_from, tack_angle)
        # Rotation à droite du vent
        right_tack_dir = self._rotate_vector(wind_from, -tack_angle)
        
        # Choisir la direction qui maximise la progression vers l'objectif (VMG)
        left_vmg = np.dot(left_tack_dir, goal_direction / (np.linalg.norm(goal_direction) + 1e-10))
        right_vmg = np.dot(right_tack_dir, goal_direction / (np.linalg.norm(goal_direction) + 1e-10))
        
        # Utiliser les laylines si disponibles
        if self.best_tack:
            if self.best_tack == "port":
                chosen_tack_dir = left_tack_dir
            else:
                chosen_tack_dir = right_tack_dir
        else:
            # Stratégie de tacking alternée pour éviter les blocages
            if self.tack_count > 5:
                # Forcer un changement de direction après un certain nombre de virements
                if self.tack_direction == 1:
                    chosen_tack_dir = left_tack_dir
                    self.tack_direction = -1
                else:
                    chosen_tack_dir = right_tack_dir
                    self.tack_direction = 1
                self.tack_count = 0
            else:
                # Choisir la direction avec le meilleur VMG
                if right_vmg > left_vmg:
                    chosen_tack_dir = right_tack_dir
                    if self.tack_direction == 1:
                        self.tack_count += 1
                    else:
                        self.tack_direction = 1
                        self.tack_count = 1
                else:
                    chosen_tack_dir = left_tack_dir
                    if self.tack_direction == -1:
                        self.tack_count += 1
                    else:
                        self.tack_direction = -1
                        self.tack_count = 1
        
        # Convertir la direction en action
        best_action = self._direction_to_action(chosen_tack_dir)
        self.last_action = best_action
        return best_action
    
    def _beam_reach_strategy(self, position: np.ndarray, velocity: np.ndarray, local_wind: np.ndarray) -> int:
        """
        Stratégie de navigation au près optimal (beam reach).
        Maintient le bateau à un angle optimal par rapport au vent pour maximiser la vitesse.
        """
        goal_direction = self.goal_position - position
        goal_norm = goal_direction / (np.linalg.norm(goal_direction) + 1e-10)
        
        # Direction du vent inversée (d'où il vient)
        wind_from = -local_wind / (np.linalg.norm(local_wind) + 1e-10)
        
        # Calculer l'angle optimal pour la vitesse maximale (environ 90° du vent)
        beam_angle = self.optimal_wind_angle
        
        # Déterminer si l'objectif est plus à gauche ou à droite du vent
        wind_goal_cross = np.cross(np.append(wind_from, 0), np.append(goal_norm, 0))[2]
        
        # Choisir la direction qui maintient l'angle optimal tout en progressant vers l'objectif
        if wind_goal_cross > 0:  # Objectif à droite du vent
            beam_dir = self._rotate_vector(wind_from, -beam_angle)
        else:  # Objectif à gauche du vent
            beam_dir = self._rotate_vector(wind_from, beam_angle)
        
        # Convertir la direction en action
        best_action = self._direction_to_action(beam_dir)
        self.last_action = best_action
        return best_action
    
    def _plan_path(self, current_position: np.ndarray) -> None:
        """
        Planifie un chemin optimal à l'aide de l'algorithme A* modifié.
        Prend en compte l'efficacité de navigation et les structures de vent favorables.
        """
        # Simplification: discrétiser la position et l'objectif
        start = tuple(current_position.astype(int))
        goal = tuple(self.goal_position.astype(int))
        
        # Initialisation des structures pour A*
        open_set = []
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}
        open_set_hash = {start}
        
        # Utiliser un heap pour l'efficacité
        heapq.heappush(open_set, (f_score[start], start))
        
        # Limiter le nombre d'itérations pour des raisons de performance
        max_iterations = 1000
        iteration = 0
        
        # Récupérer les régions favorables/défavorables si disponibles
        regions_to_avoid = set()
        favorable_regions = set()
        
        if self.wind_analysis:
            regions_to_avoid = set(self.wind_analysis.get('unfavorable_regions', []))
            favorable_regions = set(self.wind_analysis.get('favorable_regions', []))
        
        while open_set and iteration < max_iterations:
            iteration += 1
            
            # Récupérer le nœud avec le plus faible f_score
            _, current = heapq.heappop(open_set)
            open_set_hash.remove(current)
            
            # Si nous avons atteint l'objectif
            if current == goal:
                # Reconstruire le chemin
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                self.planned_path = path
                return
            
            # Explorer les voisins
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue  # Ignorer la position actuelle
                    
                    neighbor = (current[0] + dx, current[1] + dy)
                    
                    # Vérifier les limites de la grille
                    if (neighbor[0] < 0 or neighbor[0] >= self.grid_size[0] or
                        neighbor[1] < 0 or neighbor[1] >= self.grid_size[1]):
                        continue
                    
                    # Vérifier si le voisin est une position bloquée connue
                    if neighbor in self.blocked_positions:
                        continue
                    
                    # Calculer l'efficacité de navigation pour ce mouvement
                    wind_at_current = self.wind_field[current[1], current[0]]
                    direction = np.array([dx, dy])
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)
                    
                    efficiency = self._calculate_sailing_efficiency(direction, wind_at_current)
                    
                    # Coût du mouvement (distance euclidienne ajustée par l'efficacité)
                    move_cost = np.sqrt(dx*dx + dy*dy)
                    if efficiency > 0.1:
                        move_cost = move_cost / efficiency  # Coût inversement proportionnel à l'efficacité
                    else:
                        move_cost = move_cost * 10  # Pénalité pour les mouvements très inefficaces
                    
                    # Appliquer des modifications de coût basées sur l'analyse du vent
                    if neighbor in regions_to_avoid:
                        move_cost *= 2.0  # Pénaliser les zones défavorables
                    if neighbor in favorable_regions:
                        move_cost *= 0.5  # Favoriser les zones favorables
                    
                    # Calculer tentative_g_score
                    tentative_g_score = g_score[current] + move_cost
                    
                    # Si ce chemin vers le voisin est meilleur
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, goal)
                        
                        if neighbor not in open_set_hash:
                            heapq.heappush(open_set, (f_score[neighbor], neighbor))
                            open_set_hash.add(neighbor)
        
        # Si aucun chemin n'est trouvé, utiliser un chemin direct
        if iteration >= max_iterations:
            # Chemin direct simplifié (ligne droite)
            direct_path = []
            dx = goal[0] - start[0]
            dy = goal[1] - start[1]
            steps = max(abs(dx), abs(dy))
            
            if steps > 0:
                x_step = dx / steps
                y_step = dy / steps
                
                for i in range(1, min(steps + 1, 10)):  # Limiter à 10 points
                    x = start[0] + int(i * x_step)
                    y = start[1] + int(i * y_step)
                    if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
                        direct_path.append((x, y))
                
                self.planned_path = direct_path
    
    def _follow_planned_path(self, position: np.ndarray, velocity: np.ndarray, local_wind: np.ndarray) -> int:
        """Suit un chemin planifié dans le champ de vent."""
        # Vérifier si le chemin est toujours valide
        if not self.planned_path:
            self._plan_path(position)
            if not self.planned_path:
                # Si aucun chemin n'est trouvé, utiliser une autre stratégie
                return self._adaptive_movement(position, velocity, local_wind)
        
        # Trouver le prochain point du chemin
        while self.planned_path and np.linalg.norm(np.array(self.planned_path[0]) - position) < 1.5:
            self.planned_path.pop(0)
            
        if not self.planned_path:
            # Si nous avons atteint la fin du chemin, planifier un nouveau
            self._plan_path(position)
            if not self.planned_path:
                return self._adaptive_movement(position, velocity, local_wind)
        
        # Direction vers le prochain point du chemin
        next_point = np.array(self.planned_path[0])
        path_direction = next_point - position
        
        # Trouver la meilleure action pour suivre le chemin
        best_action = self._find_best_action_vmg(position, local_wind, path_direction)
        self.last_action = best_action
        return best_action
    
    def _exploration_strategy(self, position: np.ndarray, velocity: np.ndarray, local_wind: np.ndarray) -> int:
        """
        Stratégie d'exploration pour sortir des situations difficiles.
        Utilise une combinaison d'aléatoire et de direction générale vers l'objectif.
        """
        # Direction générale vers l'objectif
        goal_direction = self.goal_position - position
        
        # Calculer la distance aux bords de la grille
        dist_to_edges = [
            position[0],  # Distance au bord gauche
            self.grid_size[0] - 1 - position[0],  # Distance au bord droit
            position[1],  # Distance au bord inférieur
            self.grid_size[1] - 1 - position[1]  # Distance au bord supérieur
        ]
        min_edge_dist = min(dist_to_edges)
        
        # Si nous sommes près d'un bord, éviter de s'en approcher davantage
        if min_edge_dist < 3:
            edge_index = dist_to_edges.index(min_edge_dist)
            avoid_directions = {
                0: [6, 5, 7],  # Éviter l'ouest si près du bord gauche
                1: [2, 1, 3],  # Éviter l'est si près du bord droit
                2: [4, 3, 5],  # Éviter le sud si près du bord inférieur
                3: [0, 1, 7]   # Éviter le nord si près du bord supérieur
            }
            avoid_list = avoid_directions.get(edge_index, [])
            possible_actions = [i for i in range(8) if i not in avoid_list]
        else:
            possible_actions = list(range(8))
            
        # Mélanger les actions pour ajouter de l'aléatoire
        self.np_random.shuffle(possible_actions)
        
        # Trouver l'action qui nous rapproche le plus de l'objectif
        best_action = None
        best_vmg = -float('inf')
        
        for action in possible_actions:
            direction = self._action_to_direction(action)
            
            # Calculer la progression vers l'objectif (VMG)
            vmg = np.dot(direction, goal_direction) / (np.linalg.norm(goal_direction) + 1e-10)
            
            # Calculer l'efficacité de navigation pour cette direction
            efficiency = self._calculate_sailing_efficiency(direction, local_wind)
            
            # Combiner VMG et efficacité
            combined_score = vmg * efficiency
            
            # Ajouter un facteur aléatoire pour l'exploration
            exploration_factor = self.np_random.random() * 0.3
            combined_score += exploration_factor
            
            if combined_score > best_vmg:
                best_vmg = combined_score
                best_action = action
        
        # Si aucune action n'est trouvée, choisir une action aléatoire
        if best_action is None:
            best_action = self.np_random.integers(0, 8)
            
        self.last_action = best_action
        return best_action
    
    def _adaptive_movement(self, position: np.ndarray, velocity: np.ndarray, local_wind: np.ndarray) -> int:
        """
        Stratégie de mouvement adaptative qui combine toutes les approches.
        C'est la stratégie par défaut quand aucune autre n'est spécifiquement choisie.
        """
        goal_direction = self.goal_position - position
        
        # Direction du vent inversée (d'où il vient)
        wind_from = -local_wind / (np.linalg.norm(local_wind) + 1e-10)
        
        # Calculer l'angle entre le vent et la direction de l'objectif
        if np.linalg.norm(local_wind) > 0 and np.linalg.norm(goal_direction) > 0:
            goal_norm = goal_direction / np.linalg.norm(goal_direction)
            wind_goal_angle = np.arccos(np.clip(np.dot(wind_from, goal_norm), -1.0, 1.0))
            
            # Si l'objectif est dans la zone interdite du vent (trop près du vent)
            if wind_goal_angle < self.min_wind_angle:
                # Utiliser la stratégie de louvoyage
                return self._tacking_strategy(position, velocity, local_wind)
        
        # Par défaut, aller directement vers l'objectif
        return self._direct_to_goal(position, velocity, local_wind)
    
    def _heuristic(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Heuristique pour A* (distance euclidienne)."""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _get_action_towards_goal(self, position: np.ndarray, velocity: np.ndarray, local_wind: np.ndarray) -> int:
        """
        Obtient l'action qui nous rapproche le plus de l'objectif.
        Utilisé lorsque nous sommes très proches de l'objectif.
        """
        goal_direction = self.goal_position - position
        
        if np.linalg.norm(goal_direction) < 0.1:
            # Si nous sommes presque à l'objectif, rester sur place
            return 8
        
        # Normaliser la direction
        if np.linalg.norm(goal_direction) > 0:
            goal_direction = goal_direction / np.linalg.norm(goal_direction)
        
        # Convertir la direction en action
        best_action = self._direction_to_action(goal_direction)
        return best_action
    
    def _find_best_action_vmg(self, position: np.ndarray, local_wind: np.ndarray, target_direction: np.ndarray) -> int:
        """
        Trouve l'action qui maximise la progression vers la cible (VMG).
        
        Args:
            position: Position actuelle
            local_wind: Vent local
            target_direction: Direction cible (vers l'objectif ou un point intermédiaire)
            
        Returns:
            action: Action optimale
        """
        # Normaliser la direction cible
        if np.linalg.norm(target_direction) > 0:
            target_norm = target_direction / np.linalg.norm(target_direction)
        else:
            target_norm = np.array([0, 1])  # Par défaut vers le nord si pas de direction
        
        best_action = None
        best_score = -float('inf')
        
        # Évaluer chaque action possible
        for action in range(8):  # Toutes les directions sauf "rester sur place"
            direction = self._action_to_direction(action)
            
            # Calculer la progression vers la cible (VMG)
            vmg = np.dot(direction, target_norm)
            
            # Calculer l'efficacité de navigation pour cette direction
            efficiency = self._calculate_sailing_efficiency(direction, local_wind)
            
            # Facteur d'inertie (préférer continuer dans la même direction)
            inertia_factor = 0.0
            if self.last_action is not None and self.last_action < 8:
                last_direction = self._action_to_direction(self.last_action)
                direction_similarity = np.dot(direction, last_direction)
                inertia_factor = 0.2 * max(0, direction_similarity)
            
            # Score combiné: VMG * efficacité + inertie
            combined_score = vmg * efficiency + inertia_factor
            
            # Vérifier si cette action nous mène à une position bloquée connue
            new_position = position + direction
            new_pos_tuple = tuple(new_position.astype(int))
            if new_pos_tuple in self.blocked_positions:
                combined_score -= 0.5  # Pénalité pour les positions bloquées
            
            # Vérifier les régions favorables/défavorables
            if self.wind_analysis:
                if new_pos_tuple in self.wind_analysis.get('favorable_regions', []):
                    combined_score += 0.2  # Bonus pour les zones de vent favorable
                if new_pos_tuple in self.wind_analysis.get('unfavorable_regions', []):
                    combined_score -= 0.2  # Pénalité pour les zones de vent défavorable
            
            if combined_score > best_score:
                best_score = combined_score
                best_action = action
        
        # Si aucune action n'est trouvée ou si le meilleur score est très faible
        if best_action is None or best_score < 0.05:
            # Essayer l'action qui maximise uniquement l'efficacité
            best_action = self._find_most_efficient_action(local_wind)
        
        return best_action
    
    def _find_most_efficient_action(self, local_wind: np.ndarray) -> int:
        """
        Trouve l'action qui offre la meilleure efficacité de navigation,
        indépendamment de la direction vers l'objectif.
        """
        best_action = None
        best_efficiency = -1
        
        for action in range(8):  # Toutes les directions sauf "rester sur place"
            direction = self._action_to_direction(action)
            efficiency = self._calculate_sailing_efficiency(direction, local_wind)
            
            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_action = action
        
        return best_action if best_action is not None else 0  # Par défaut vers le nord
    
    def _calculate_sailing_efficiency(self, boat_direction: np.ndarray, wind: np.ndarray) -> float:
        """
        Calcule l'efficacité de navigation basée sur l'angle entre la direction du bateau et le vent.
        
        Args:
            boat_direction: Vecteur normalisé de la direction du bateau
            wind: Vecteur du vent (direction où il va)
            
        Returns:
            sailing_efficiency: Flottant entre 0.05 et 1.0 représentant l'efficacité
        """
        # Vérifier si les directions sont normalisées
        if np.linalg.norm(boat_direction) < 1e-10:
            return 0.05  # Valeur minimale par défaut
        
        # Normaliser si nécessaire
        boat_dir_norm = boat_direction / np.linalg.norm(boat_direction)
        
        # Direction d'où vient le vent
        if np.linalg.norm(wind) < 1e-10:
            return 1.0  # Pas de vent, efficacité maximale (on peut aller n'importe où)
        
        wind_from = -wind / np.linalg.norm(wind)
        
        # Créer une clé de cache pour ce calcul
        cache_key = (tuple(boat_dir_norm.round(3)), tuple(wind_from.round(3)))
        
        # Vérifier si le résultat est déjà dans le cache
        if cache_key in self.efficiency_cache:
            return self.efficiency_cache[cache_key]
        
        # Calculer l'angle entre le vent et la direction
        wind_angle = np.arccos(np.clip(np.dot(wind_from, boat_dir_norm), -1.0, 1.0))
        
        # Calculer l'efficacité de navigation basée sur l'angle au vent
        if wind_angle < np.pi/4:  # Moins de 45 degrés au vent
            sailing_efficiency = 0.05  # Petite mais non nulle efficacité dans la zone interdite
        elif wind_angle < np.pi/2:  # Entre 45 et 90 degrés
            sailing_efficiency = 0.5 + 0.5 * (wind_angle - np.pi/4) / (np.pi/4)  # Augmentation linéaire jusqu'à 1.0
        elif wind_angle < 3*np.pi/4:  # Entre 90 et 135 degrés
            sailing_efficiency = 1.0  # Efficacité maximale
        else:  # Plus de 135 degrés
            sailing_efficiency = 1.0 - 0.5 * (wind_angle - 3*np.pi/4) / (np.pi/4)  # Diminution linéaire
            sailing_efficiency = max(0.5, sailing_efficiency)  # Mais toujours décente
        
        # Stocker dans le cache
        self.efficiency_cache[cache_key] = sailing_efficiency
        
        return sailing_efficiency
    
    def _action_to_direction(self, action: int) -> np.ndarray:
        """Convertit un indice d'action en vecteur de direction."""
        # Correspondance des actions aux vecteurs de direction
        directions = [
            (0, 1),     # 0: Nord (Y croissant)
            (1, 1),     # 1: Nord-Est
            (1, 0),     # 2: Est (X croissant)
            (1, -1),    # 3: Sud-Est
            (0, -1),    # 4: Sud (Y décroissant)
            (-1, -1),   # 5: Sud-Ouest
            (-1, 0),    # 6: Ouest (X décroissant)
            (-1, 1),    # 7: Nord-Ouest
            (0, 0)      # 8: Rester sur place
        ]
        
        # Normaliser les directions diagonales
        direction = np.array(directions[action])
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
        
        return direction
    
    def _direction_to_action(self, direction: np.ndarray) -> int:
        """
        Convertit un vecteur de direction en action.
        Trouve l'action dont la direction est la plus proche du vecteur donné.
        """
        if np.linalg.norm(direction) < 1e-10:
            return 8  # Rester sur place si pas de direction
        
        # Normaliser la direction
        direction_norm = direction / np.linalg.norm(direction)
        
        # Calculer l'angle avec le Nord (0, 1)
        north = np.array([0, 1])
        angle = np.arccos(np.clip(np.dot(north, direction_norm), -1.0, 1.0))
        
        # Déterminer le côté (est ou ouest)
        east_side = direction_norm[0] >= 0
        
        # Mapper l'angle à l'une des 8 directions
        if angle < np.pi/8:
            return 0  # Nord
        elif angle < 3*np.pi/8:
            return 1 if east_side else 7  # Nord-Est ou Nord-Ouest
        elif angle < 5*np.pi/8:
            return 2 if east_side else 6  # Est ou Ouest
        elif angle < 7*np.pi/8:
            return 3 if east_side else 5  # Sud-Est ou Sud-Ouest
        else:
            return 4  # Sud
    
    def _rotate_vector(self, vector: np.ndarray, angle: float) -> np.ndarray:
        """
        Tourne un vecteur 2D d'un angle donné (en radians).
        
        Args:
            vector: Vecteur à tourner
            angle: Angle de rotation en radians (positif = sens antihoraire)
            
        Returns:
            Vecteur tourné
        """
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        # Matrice de rotation
        rot_x = vector[0] * cos_angle - vector[1] * sin_angle
        rot_y = vector[0] * sin_angle + vector[1] * cos_angle
        
        return np.array([rot_x, rot_y])