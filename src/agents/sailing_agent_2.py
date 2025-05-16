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
        Calcule l'efficacité de navigation basée sur une polaire de vitesse plus réaliste.
        """
        if np.linalg.norm(boat_direction_norm) < 0.001 or np.linalg.norm(wind_from_direction_norm) < 0.001:
            return 0.05  # Efficacité minimale si pas de mouvement ou pas de vent

        cos_angle = np.dot(wind_from_direction_norm, boat_direction_norm)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        wind_angle = np.arccos(cos_angle)  # Angle entre le bateau et d'où vient le vent

        # Polaire de vitesse plus réaliste basée sur des données empiriques
        # Ces valeurs pourraient être chargées depuis un fichier externe ou une table
        wind_angles_rad = np.array([0, 0.52, 0.87, 1.57, 2.09, 2.62, 3.14])  # en radians (0°, 30°, 50°, 90°, 120°, 150°, 180°)
        efficiency_values = np.array([0.0, 0.3, 0.6, 1.0, 0.9, 0.7, 0.6])    # efficacité correspondante
        
        # Interpolation pour obtenir l'efficacité à l'angle spécifique
        return np.interp(wind_angle, wind_angles_rad, efficiency_values)


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
        Détecte des couloirs de vent et des gradients favorables dans le champ de vent local.
        """
        if wind_field_flat is None or len(wind_field_flat) == 0:
            return []
            
        wind_field_2d = wind_field_flat.reshape(self.grid_size[1], self.grid_size[0], 2)
        favorable_directions = []
        
        # Rayon d'analyse (plus grand pour une meilleure analyse)
        radius = 3
        x, y = int(position[0]), int(position[1])
        
        # Calculer le gradient du vent (force du vent) dans la zone locale
        wind_magnitudes = np.zeros((2*radius+1, 2*radius+1))
        wind_directions = np.zeros((2*radius+1, 2*radius+1, 2))
        
        for i in range(-radius, radius+1):
            for j in range(-radius, radius+1):
                check_x, check_y = x + i, y + j
                if 0 <= check_x < self.grid_size[0] and 0 <= check_y < self.grid_size[1]:
                    wind_vec = wind_field_2d[check_y, check_x]
                    wind_magnitudes[i+radius, j+radius] = np.linalg.norm(wind_vec)
                    if np.linalg.norm(wind_vec) > 0.001:
                        wind_directions[i+radius, j+radius] = wind_vec / np.linalg.norm(wind_vec)
        
        # Détecter les gradients de vent (zones où la force du vent augmente)
        gradient_x = np.zeros_like(wind_magnitudes)
        gradient_y = np.zeros_like(wind_magnitudes)
        
        # Calculer gradient simple (différence avec cases adjacentes)
        for i in range(1, 2*radius):
            for j in range(1, 2*radius):
                gradient_x[i, j] = wind_magnitudes[i+1, j] - wind_magnitudes[i-1, j]
                gradient_y[i, j] = wind_magnitudes[i, j+1] - wind_magnitudes[i, j-1]
        
        # Identifier les directions avec fort gradient positif (vers des vents plus forts)
        for i in range(1, 2*radius):
            for j in range(1, 2*radius):
                if abs(gradient_x[i, j]) > 0.2 or abs(gradient_y[i, j]) > 0.2:  # Seuil de gradient significatif
                    # Direction du gradient
                    gradient_dir = np.array([gradient_x[i, j], gradient_y[i, j]])
                    if np.linalg.norm(gradient_dir) > 0.001:
                        gradient_dir = gradient_dir / np.linalg.norm(gradient_dir)
                        
                        # Vérifier si cette direction est globalement vers le but
                        if np.dot(gradient_dir, goal_direction_norm) > 0.3:
                            # Vérifier si le vent à destination sera favorable pour naviguer
                            dest_x, dest_y = x + int(gradient_dir[0] * radius), y + int(gradient_dir[1] * radius)
                            if 0 <= dest_x < self.grid_size[0] and 0 <= dest_y < self.grid_size[1]:
                                dest_wind = wind_field_2d[dest_y, dest_x]
                                dest_wind_from = self._get_wind_from_direction_normalized(dest_wind)
                                
                                # Calculer l'efficacité potentielle avec ce vent
                                eff = self.calculate_sailing_efficiency(gradient_dir, dest_wind_from)
                                if eff > 0.6:  # Bonne efficacité attendue
                                    # Trouver la direction discrète la plus proche pour l'action
                                    best_action_idx = self.direction_to_action(gradient_dir)
                                    favorable_directions.append(self.ACTION_VECTORS[best_action_idx])
        
        return favorable_directions


    def is_position_valid(self, position: np.ndarray) -> bool:
        """Vérifie si une position est dans les limites de la grille."""
        return (0 <= position[0] < self.grid_size[0] and
                0 <= position[1] < self.grid_size[1])

    def _update_progress_and_strategy(self, current_distance_to_goal: float, position: np.ndarray, wind: np.ndarray):
        """
        Met à jour le suivi de la progression et choisit la stratégie la plus adaptée
        en fonction de multiples facteurs analysés.
        """
        # Analyser la progression comme avant
        made_progress = False
        if self.previous_position is not None:
            pos_change = np.linalg.norm(position - self.previous_position)
            dist_improvement = self.last_distance_to_goal - current_distance_to_goal

            if dist_improvement >= self.MIN_PROGRESS_THRESHOLD or \
            (dist_improvement > -0.1 and pos_change > 0.1):
                self.steps_without_progress = 0
                made_progress = True
            else:
                self.steps_without_progress += 1
        else:
            self.steps_without_progress = 0
            made_progress = True

        # Système de scoring pour chaque stratégie
        scores = {
            "direct": 0,
            "tacking": 0,
            "exploration": 0
        }
        
        # Facteurs pour le score "direct"
        # 1. Distance au but
        if current_distance_to_goal < 5.0:
            scores["direct"] += 2  # Privilégier direct quand on est proche
        
        # 2. Vent favorable pour route directe
        goal_vector = self.goal_position - position
        goal_dir_norm = goal_vector / (np.linalg.norm(goal_vector) + 1e-6)
        wind_from_norm = self._get_wind_from_direction_normalized(wind)
        
        direct_efficiency = self.calculate_sailing_efficiency(goal_dir_norm, wind_from_norm)
        scores["direct"] += direct_efficiency * 5  # 0-5 points basés sur l'efficacité
        
        # Facteurs pour le score "tacking"
        # 1. Vent défavorable pour route directe (près du vent)
        if direct_efficiency < 0.3:  # Mauvaise efficacité en route directe
            scores["tacking"] += 4
            
        # 2. Distance significative à parcourir
        if current_distance_to_goal > 5.0:
            scores["tacking"] += 2  # Tacking plus utile sur longue distance
        
        # Facteurs pour le score "exploration"
        # 1. Manque de progrès prolongé
        if self.steps_without_progress > self.max_stuck_steps / 2:
            scores["exploration"] += 3
        
        # 2. Zone avec vent faible
        if np.linalg.norm(wind) < 0.3:  # Vent faible
            scores["exploration"] += 3
        
        # 3. Si les autres stratégies ont échoué précédemment
        if not made_progress and self.strategy in ["direct", "tacking"]:
            scores["exploration"] += 2
        
        # Malus si on vient juste d'utiliser une stratégie sans succès
        if not made_progress:
            scores[self.strategy] -= 2
        
        # Choisir la stratégie avec le meilleur score
        best_strategy = max(scores.items(), key=lambda x: x[1])[0]
        
        # Ne changer que si la différence de score est significative ou si on est bloqué
        if scores[best_strategy] > scores[self.strategy] + 1 or \
        self.steps_without_progress >= self.max_stuck_steps:
            if best_strategy != self.strategy:
                # print(f"Changing strategy from {self.strategy} to {best_strategy}. Scores: {scores}")
                self.strategy = best_strategy
                # Réinitialiser les compteurs spécifiques à la stratégie
                if best_strategy == "tacking":
                    self.tack_counter = 0
                elif best_strategy == "exploration":
                    # Augmenter temporairement l'exploration
                    self.exploration_rate = min(self.DEFAULT_EXPLORATION_RATE * 2, 0.2)
                elif best_strategy == "direct":
                    # Rétablir l'exploration normale
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