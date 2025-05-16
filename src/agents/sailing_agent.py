import numpy as np
from collections import deque
from agents.base_agent import BaseAgent # Assurez-vous que ce chemin est correct

class RuleBasedSailor(BaseAgent):
    """
    Agent basé sur des règles pour le Sailing Challenge.
    Utilise des connaissances de navigation à voile et des heuristiques simples
    pour naviguer efficacement vers un objectif.
    """

    # Paramètres de directions (pré-calculés pour l'efficacité)
    _BASE_DIRECTIONS_RAW = [
        (0, 1), (1, 1), (1, 0), (1, -1), (0, -1),
        (-1, -1), (-1, 0), (-1, 1), (0, 0)
    ]
    ACTION_VECTORS = [np.array(d) for d in _BASE_DIRECTIONS_RAW]
    NORMALIZED_MOVEMENT_VECTORS = [
        v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else v
        for v in ACTION_VECTORS[:8] # Exclure (0,0)
    ]

    # Angles de navigation (en degrés pour la lisibilité)
    OPTIMAL_BEAM_REACH_ANGLE_DEG = 90.0
    CLOSE_HAULED_ANGLE_DEG = 45.0
    NO_GO_ZONE_ANGLE_DEG = 30.0 # Zone où la progression est quasi nulle voire négative

    # Paramètres de stratégie
    DEFAULT_EXPLORATION_RATE = 0.05
    DEFAULT_MAX_STUCK_STEPS = 10 # Légèrement augmenté
    MIN_PROGRESS_THRESHOLD = 0.05 # Seuil pour considérer une progression
    DEFAULT_TACK_DURATION = 8 # Nombre de pas sur une amure avant d'envisager un virement

    def __init__(self, grid_size=(32, 32), goal_position=None):
        super().__init__()
        self.np_random = np.random.default_rng()

        self.grid_size = np.array(grid_size)
        if goal_position is None:
            self.goal_position = np.array([self.grid_size[0] // 2, self.grid_size[1] - 1])
        else:
            self.goal_position = np.array(goal_position)

        # Conversion des angles en radians
        self.optimal_beam_reach_angle = np.deg2rad(self.OPTIMAL_BEAM_REACH_ANGLE_DEG)
        self.close_hauled_angle = np.deg2rad(self.CLOSE_HAULED_ANGLE_DEG)
        self.no_go_zone_angle = np.deg2rad(self.NO_GO_ZONE_ANGLE_DEG)

        # État interne
        self.previous_position = None
        self.previous_action = 8 # Initialiser à "rester sur place"
        self.steps_without_progress = 0
        self.last_distance_to_goal = float('inf')
        self.current_tack_direction = 1  # 1 pour tribord (route à droite du vent), -1 pour bâbord
        self.tack_counter = 0
        self.tack_duration = self.DEFAULT_TACK_DURATION

        self.wind_memory = deque(maxlen=5) # Pour lisser/analyser le vent

        # Comportements et stratégies
        self.exploration_rate = self.DEFAULT_EXPLORATION_RATE
        self.max_stuck_steps = self.DEFAULT_MAX_STUCK_STEPS
        self.strategy = "direct" # Stratégies possibles: "direct", "tacking", "exploration"

    def action_to_direction(self, action: int) -> np.ndarray:
        """Convertit l'indice d'action en vecteur de direction."""
        return self.ACTION_VECTORS[action]

    def direction_to_action(self, direction_vector: np.ndarray) -> int:
        """Convertit un vecteur de direction en indice d'action le plus proche."""
        if np.linalg.norm(direction_vector) < 0.01: # Si vecteur quasi nul
            return 8 # Rester en place

        # Normaliser la direction souhaitée
        norm_target_direction = direction_vector / np.linalg.norm(direction_vector)

        best_match_idx = 8 # Par défaut: rester en place
        highest_similarity = -np.inf

        for i, norm_candidate_dir in enumerate(self.NORMALIZED_MOVEMENT_VECTORS):
            similarity = np.dot(norm_target_direction, norm_candidate_dir)
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match_idx = i
        return best_match_idx

    def _get_wind_from_direction_normalized(self, wind_to_direction: np.ndarray) -> np.ndarray:
        """Calcule la direction normalisée D'OÙ vient le vent."""
        norm = np.linalg.norm(wind_to_direction)
        if norm < 0.001:
            return np.array([0.0, 0.0]) # Pas de vent défini
        return -wind_to_direction / norm

    def calculate_sailing_efficiency(self, boat_direction_norm: np.ndarray, wind_from_direction_norm: np.ndarray) -> float:
        """
        Calcule l'efficacité de navigation.
        Args:
            boat_direction_norm: Vecteur normalisé de la direction souhaitée du bateau.
            wind_from_direction_norm: Vecteur normalisé de la direction D'OÙ vient le vent.
        """
        if np.linalg.norm(boat_direction_norm) < 0.001 or np.linalg.norm(wind_from_direction_norm) < 0.001:
            return 0.05 # Efficacité minimale si pas de mouvement ou pas de vent

        cos_angle = np.dot(wind_from_direction_norm, boat_direction_norm)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        wind_angle = np.arccos(cos_angle) # Angle entre le bateau et d'où vient le vent

        # Logique d'efficacité (ajustée pour plus de clarté sur les interpolations)
        if wind_angle <= self.no_go_zone_angle:
            return 0.05 # Dans la "no-go zone"
        elif wind_angle <= self.close_hauled_angle: # De no-go à près serré
            progress = (wind_angle - self.no_go_zone_angle) / (self.close_hauled_angle - self.no_go_zone_angle)
            return 0.05 + progress * (0.5 - 0.05) # Efficacité de 0.05 à 0.5 (près serré)
        elif wind_angle <= self.optimal_beam_reach_angle: # De près serré à travers
            progress = (wind_angle - self.close_hauled_angle) / (self.optimal_beam_reach_angle - self.close_hauled_angle)
            return 0.5 + progress * (1.0 - 0.5) # Efficacité de 0.5 à 1.0 (bon plein à travers)
        elif wind_angle <= np.pi * 0.75:  # 135 degrés, vent de travers à largue
            return 1.0 # Efficacité maximale
        else:  # Plus de 135 degrés, grand largue à vent arrière
            # Diminution de 1.0 à (par exemple) 0.7 pour le vent arrière direct
            progress = (wind_angle - np.pi * 0.75) / (np.pi - np.pi * 0.75)
            return max(0.7, 1.0 - progress * (1.0 - 0.7))


    def find_best_sailing_direction(self, position: np.ndarray, current_wind_to: np.ndarray, goal_vector: np.ndarray) -> tuple[np.ndarray, int]:
        """
        Trouve la meilleure direction de navigation discrète (action).
        """
        wind_from_norm = self._get_wind_from_direction_normalized(current_wind_to)
        goal_dist = np.linalg.norm(goal_vector)

        if goal_dist < 0.1: # Arbitrairement proche de l'objectif
            return np.array([0,0]), 8 # Rester sur place

        goal_direction_norm = goal_vector / goal_dist

        if np.linalg.norm(wind_from_norm) < 0.001: # Pas de vent
            return goal_direction_norm, self.direction_to_action(goal_direction_norm)

        best_action = 8 # Par défaut rester sur place
        best_score = -np.inf

        # Tester les 8 directions de mouvement possibles
        for action_idx, boat_dir_norm in enumerate(self.NORMALIZED_MOVEMENT_VECTORS):
            efficiency = self.calculate_sailing_efficiency(boat_dir_norm, wind_from_norm)
            
            # Projection de la direction du bateau sur la direction de l'objectif
            # Donne une mesure de "progression vers l'objectif" si cette direction est suivie
            progress_towards_goal = np.dot(boat_dir_norm, goal_direction_norm)

            # Score: on veut une bonne efficacité ET une progression vers le but.
            # Max(0, progress_towards_goal) évite les scores négatifs si on s'éloigne.
            # Le facteur 0.1 + .. évite que le score soit nul si parfaitement perpendiculaire mais nécessaire.
            score = efficiency * (0.1 + max(0, progress_towards_goal))
            
            if score > best_score:
                best_score = score
                best_action = action_idx
        
        # Optionnel : vérifier si aller directement vers le but est viable
        # direct_boat_eff = self.calculate_sailing_efficiency(goal_direction_norm, wind_from_norm)
        # if direct_boat_eff > 0.8 and best_score < direct_boat_eff * (0.1 + 1.0) :
        #     # Si la route directe est très bonne et meilleure que les options discrètes.
        #     # Cela pourrait être redondant si les NORMALIZED_MOVEMENT_VECTORS sont assez denses.
        #     # Pour l'instant, on se fie aux 8 directions discrètes.
        #     pass

        # Si la meilleure efficacité trouvée pour une action est très faible (ex: face au vent),
        # la logique de changement de stratégie dans `act` devrait prendre le relais (passer en "tacking").
        if self.calculate_sailing_efficiency(self.NORMALIZED_MOVEMENT_VECTORS[best_action], wind_from_norm) < 0.2: # seuil bas pour l'efficacité
             # Indique que même la meilleure option directe n'est pas bonne.
             # Laisser la gestion de stratégie dans `act` décider s'il faut louvoyer.
             pass # On retourne quand même la "moins pire" action directe.

        return self.ACTION_VECTORS[best_action], best_action


    def get_tacking_action(self, current_wind_to: np.ndarray, goal_vector: np.ndarray) -> int:
        """
        Détermine l'action de louvoyage.
        """
        wind_from_norm = self._get_wind_from_direction_normalized(current_wind_to)
        goal_dist = np.linalg.norm(goal_vector)
        if goal_dist < 0.1 or np.linalg.norm(wind_from_norm) < 0.001:
            return self.direction_to_action(goal_vector) # Aller au but si proche ou pas de vent

        goal_direction_norm = goal_vector / goal_dist
        
        self.tack_counter += 1
        if self.tack_counter >= self.tack_duration:
            # Pourrait être plus intelligent : vérifier si on croise l'axe du vent par rapport au but
            self.current_tack_direction *= -1 # Changer d'amure
            self.tack_counter = 0
            # print(f"Tacking: Changed tack to {'starboard' if self.current_tack_direction == 1 else 'port'}")

        # Angle d'où vient le vent
        wind_from_angle_rad = np.arctan2(wind_from_norm[1], wind_from_norm[0])
        
        # Calculer l'angle de navigation au près serré sur l'amure actuelle
        # self.current_tack_direction = 1 (tribord amures, vent vient de bâbord, on va à droite du vent)
        # self.current_tack_direction = -1 (bâbord amures, vent vient de tribord, on va à gauche du vent)
        tack_boat_angle_rad = wind_from_angle_rad + self.close_hauled_angle * self.current_tack_direction
        
        tack_direction_vector = np.array([np.cos(tack_boat_angle_rad), np.sin(tack_boat_angle_rad)])
        return self.direction_to_action(tack_direction_vector)

    def detect_local_wind_patterns(self, position: np.ndarray, wind_field_flat: np.ndarray, goal_direction_norm: np.ndarray) -> list[np.ndarray]:
        """
        Détecte des directions potentiellement intéressantes basées sur le champ de vent local.
        Retourne une liste de vecteurs de direction (non normalisés).
        """
        if wind_field_flat is None or len(wind_field_flat) == 0:
            return []
            
        wind_field_2d = wind_field_flat.reshape(self.grid_size[1], self.grid_size[0], 2)
        
        favorable_directions = []
        
        # Analyser une petite zone autour du bateau (ex: +/- 2 cases)
        # et voir si des vents plus forts/mieux orientés sont détectables
        # Cette fonction est un placeholder pour une analyse plus poussée.
        # L'idée est de chercher des "couloirs de vent" ou des zones où le vent est plus fort
        # et orienté de manière à pouvoir être utilisé.
        
        # Exemple simple : vérifier les 8 directions si le vent y est plus fort
        # et si ce vent permet de naviguer dans cette direction.
        for action_idx, move_dir_norm in enumerate(self.NORMALIZED_MOVEMENT_VECTORS):
            check_pos = np.round(position + move_dir_norm * 2).astype(int) # Regarder 2 cases plus loin

            if self.is_position_valid(check_pos):
                wind_at_check_pos = wind_field_2d[check_pos[1], check_pos[0]]
                wind_at_check_pos_from_norm = self._get_wind_from_direction_normalized(wind_at_check_pos)
                
                # Si le vent à ce point est significativement plus fort
                # ET si on peut naviguer dans la direction `move_dir_norm` avec ce vent
                current_wind_norm = np.linalg.norm(self.wind_memory[-1] if self.wind_memory else np.array([0,0]))
                
                if np.linalg.norm(wind_at_check_pos) > current_wind_norm * 1.2: # Vent 20% plus fort
                    eff = self.calculate_sailing_efficiency(move_dir_norm, wind_at_check_pos_from_norm)
                    if eff > 0.6: # Et on peut l'utiliser efficacement
                        # Et cette direction est globalement vers le but
                        if np.dot(move_dir_norm, goal_direction_norm) > 0.3: # Au moins un peu vers le but
                             favorable_directions.append(self.ACTION_VECTORS[action_idx])
                             
        return favorable_directions


    def is_position_valid(self, position: np.ndarray) -> bool:
        """Vérifie si une position est dans les limites de la grille."""
        return (0 <= position[0] < self.grid_size[0] and
                0 <= position[1] < self.grid_size[1])

    def _update_progress_and_strategy(self, current_distance_to_goal: float, position: np.ndarray):
        """Met à jour le suivi de la progression et change de stratégie si bloqué."""
        made_progress = False
        if self.previous_position is not None:
            # On peut aussi vérifier si la position a changé, en plus de la distance au but
            # car on peut être bloqué contre un mur sans que la distance au but change beaucoup.
            pos_change = np.linalg.norm(position - self.previous_position)
            dist_improvement = self.last_distance_to_goal - current_distance_to_goal

            if dist_improvement >= self.MIN_PROGRESS_THRESHOLD or \
               (dist_improvement > -0.1 and pos_change > 0.1) : # On a bougé et pas trop reculé
                self.steps_without_progress = 0
                made_progress = True
            else:
                self.steps_without_progress += 1
        else: # Premier pas
             self.steps_without_progress = 0
             made_progress = True


        if self.steps_without_progress >= self.max_stuck_steps:
            # print(f"Stuck at {position} with strategy {self.strategy}. Steps w/o progress: {self.steps_without_progress}. Dist to goal: {current_distance_to_goal}")
            self.steps_without_progress = 0 # Réinitialiser pour la nouvelle stratégie
            if self.strategy == "direct":
                self.strategy = "tacking"
                self.tack_counter = 0 # Forcer un nouveau calcul d'amure
                self.current_tack_direction *= -1 # Essayer l'autre amure en premier
            elif self.strategy == "tacking":
                self.strategy = "exploration" # Essayer de trouver un meilleur vent
            elif self.strategy == "exploration":
                # Si l'exploration échoue, on pourrait tenter un mouvement aléatoire plus franc
                # ou simplement revenir à direct.
                self.strategy = "direct"
                self.exploration_rate = min(self.DEFAULT_EXPLORATION_RATE * 2, 0.2) # Augmenter temporairement l'exploration
            # print(f"New strategy: {self.strategy}")
        elif made_progress and self.strategy == "exploration": # Si l'exploration a fonctionné
            self.strategy = "direct" # Revenir à direct
            self.exploration_rate = self.DEFAULT_EXPLORATION_RATE # Rétablir l'exploration normale
            # print("Exploration successful, back to direct.")
        elif made_progress:
            self.exploration_rate = self.DEFAULT_EXPLORATION_RATE


    def _get_action_based_on_strategy(self, position: np.ndarray, smoothed_wind: np.ndarray, goal_vector: np.ndarray, wind_field_flat: np.ndarray) -> int:
        """Choisit une action en fonction de la stratégie."""
        goal_norm = np.linalg.norm(goal_vector)
        goal_direction_norm = goal_vector / goal_norm if goal_norm > 0.01 else np.array([0,0])

        action = 8 # Default: stay
        if self.strategy == "direct":
            _, action = self.find_best_sailing_direction(position, smoothed_wind, goal_vector)
        elif self.strategy == "tacking":
            action = self.get_tacking_action(smoothed_wind, goal_vector)
        elif self.strategy == "exploration":
            favorable_wind_dirs = self.detect_local_wind_patterns(position, wind_field_flat, goal_direction_norm)
            if favorable_wind_dirs:
                # Choisir la direction la plus alignée avec l'objectif ou la plus forte
                best_explore_dir = None
                best_align_score = -np.inf
                for dir_vec in favorable_wind_dirs:
                    align = np.dot(dir_vec / (np.linalg.norm(dir_vec) + 1e-6), goal_direction_norm)
                    if align > best_align_score:
                        best_align_score = align
                        best_explore_dir = dir_vec
                if best_explore_dir is not None:
                    action = self.direction_to_action(best_explore_dir)
                else: # Aucun vent favorable trouvé, passer en direct
                    self.strategy = "direct"
                    _, action = self.find_best_sailing_direction(position, smoothed_wind, goal_vector)
            else: # Pas de motif clair, revenir à direct (ou tenter une action aléatoire de déblocage)
                # print("Exploration found no specific patterns, trying direct strategy.")
                self.strategy = "direct"
                _, action = self.find_best_sailing_direction(position, smoothed_wind, goal_vector)
        return action

    def _ensure_action_validity_and_avoid_oscillation(self, position: np.ndarray, chosen_action: int, goal_vector: np.ndarray) -> int:
        """Vérifie la validité de l'action et tente d'éviter les oscillations simples."""
        final_action = chosen_action

        # 1. Vérifier la validité par rapport aux bords de la grille
        next_pos_candidate = position + self.action_to_direction(final_action)
        if not self.is_position_valid(next_pos_candidate):
            # print(f"Action {final_action} from {position} to {next_pos_candidate} is invalid (out of bounds). Finding alternative.")
            # Essayer des actions alternatives, en privilégiant celles qui ne s'éloignent pas trop du but
            best_alt_action = 8 # Rester en place si rien d'autre
            best_alt_score = -np.inf
            goal_direction_norm = goal_vector / (np.linalg.norm(goal_vector) + 1e-6)

            for alt_idx in range(8): # Tester les 8 mouvements
                alt_next_pos = position + self.action_to_direction(alt_idx)
                if self.is_position_valid(alt_next_pos):
                    # Score basé sur l'alignement avec le but
                    action_dir_norm = self.NORMALIZED_MOVEMENT_VECTORS[alt_idx]
                    score = np.dot(action_dir_norm, goal_direction_norm)
                    if score > best_alt_score:
                        best_alt_score = score
                        best_alt_action = alt_idx
            final_action = best_alt_action
            # print(f"Alternative action chosen: {final_action}")

        # 2. Anti-oscillation simple : si on est sur le point de répéter l'inverse de l'action précédente
        #    et qu'on n'a pas progressé, essayer autre chose.
        #    (Ex: si action précédente était Nord (0), et action actuelle est Sud (4))
        if self.previous_action < 8 and final_action < 8: # Si ce sont des mouvements
            if (self.previous_action + 4) % 8 == final_action: # Action opposée
                if self.steps_without_progress > 1: # Et on est un peu bloqué
                    # print(f"Potential oscillation detected: prev={self.previous_action}, current={final_action}. Trying to break.")
                    # Tenter une action aléatoire différente ou une action adjacente
                    # non opposée à la précédente ni l'action courante.
                    possible_actions = list(range(8))
                    possible_actions.remove(final_action)
                    if self.previous_action in possible_actions:
                        possible_actions.remove(self.previous_action)
                    
                    if possible_actions:
                        new_action_candidate = self.np_random.choice(possible_actions)
                        # Vérifier à nouveau la validité de cette nouvelle action
                        if self.is_position_valid(position + self.action_to_direction(new_action_candidate)):
                            final_action = new_action_candidate
                            # print(f"Breaking oscillation with action: {final_action}")
                        # else, on garde l'action `final_action` calculée avant l'anti-oscillation (après validité bord)
                    # Si aucune autre action n'est possible (rare), on garde final_action
        return final_action


    def act(self, observation: list) -> int:
        position = np.array([observation[0], observation[1]])
        # velocity = np.array([observation[2], observation[3]]) # Non utilisé actuellement
        raw_wind_at_position = np.array([observation[4], observation[5]])
        wind_field_flat = np.array(observation[6:]) if len(observation) > 6 else None

        # Lissage du vent
        self.wind_memory.append(raw_wind_at_position)
        if len(self.wind_memory) >= 3: # Utiliser une moyenne sur quelques pas
            smoothed_wind = np.mean(np.array(list(self.wind_memory)), axis=0)
        else:
            smoothed_wind = raw_wind_at_position

        goal_vector = self.goal_position - position
        current_distance_to_goal = np.linalg.norm(goal_vector)

        if current_distance_to_goal < 0.5: # Seuil pour considérer l'objectif atteint
            # print("Goal reached!")
            self.previous_action = 8
            return 8 # Rester en place

        # Mise à jour de la progression et potentiellement de la stratégie
        self._update_progress_and_strategy(current_distance_to_goal, position)

        # Décision de l'action
        action_chosen: int
        if self.np_random.random() < self.exploration_rate and self.strategy != "exploration":
            action_chosen = self.np_random.integers(0, 8) # Mouvement aléatoire (0-7)
        else:
            if np.linalg.norm(smoothed_wind) < 0.001: # Pas de vent significatif
                action_chosen = self.direction_to_action(goal_vector) # Aller direct au but
            else:
                action_chosen = self._get_action_based_on_strategy(
                    position, smoothed_wind, goal_vector, wind_field_flat
                )

        # Validation finale de l'action et anti-oscillation
        final_action = self._ensure_action_validity_and_avoid_oscillation(position, action_chosen, goal_vector)
        
        # Sauvegarde de l'état pour le prochain pas
        self.previous_position = position.copy()
        self.last_distance_to_goal = current_distance_to_goal
        self.previous_action = final_action
        
        # print(f"Pos: {position}, GoalDist: {current_distance_to_goal:.2f}, Wind: {smoothed_wind}, Strat: {self.strategy}, Action: {final_action}")
        return final_action