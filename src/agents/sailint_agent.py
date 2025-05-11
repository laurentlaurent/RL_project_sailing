"""
WindAwareNavigator - Agent de navigation à voile avancé pour le Sailing Challenge

Cet agent combine:
1. Compréhension de la physique de la voile (points de navigation optimaux)
2. Planification de trajectoire en fonction du champ de vent complet
3. Apprentissage par renforcement avec état augmenté
4. Stratégies adaptatives basées sur les configurations de vent

L'agent est capable de:
- Analyser le champ de vent complet pour planifier des itinéraires
- Naviguer efficacement contre le vent (louvoyer)
- S'adapter à l'évolution des conditions de vent
- Choisir les meilleurs angles par rapport au vent pour maximiser la vitesse
"""

import numpy as np
from agents.base_agent import BaseAgent

class WindAwareNavigator(BaseAgent):
    """
    Agent avancé pour le Sailing Challenge qui utilise la compréhension de la physique de la voile 
    et une approche hybride combinant planification et apprentissage par renforcement.
    """
    
    def __init__(self):
        """Initialise l'agent avec les paramètres nécessaires."""
        super().__init__()
        self.np_random = np.random.default_rng()
        
        # Paramètres de l'environnement
        self.grid_size = (32, 32)
        self.goal_position = np.array([self.grid_size[0] // 2, self.grid_size[1] - 1])
        
        # Paramètres de discrétisation pour l'état
        self.position_bins = 8
        self.velocity_bins = 6
        self.wind_bins = 8
        
        # Paramètres de la physique de la voile
        self.optimal_beam_reach_angle = np.pi/2  # 90 degrés, perpendiculaire au vent
        self.close_hauled_angle = np.pi/4        # 45 degrés, près du vent
        self.no_go_zone_angle = np.pi/6          # 30 degrés, zone impossible
        
        # Table Q pour l'apprentissage par renforcement
        self.q_table = {}
        self._init_q_table()
        
        # Paramètres de la stratégie
        self.exploration_rate = 0.05  # Faible taux pour l'exploitation en phase de test
        self.use_wind_planning = True
        self.use_tacking = True
        self.current_tack_direction = 1  # 1 pour tribord, -1 pour bâbord
        self.patience = 0
        
        # État interne
        self.previous_position = None
        self.steps_without_progress = 0
        self.last_distance_to_goal = float('inf')
        
    def _init_q_table(self):
        """Initialise la table Q avec quelques valeurs pré-calculées basées sur la physique de la voile."""
        # Points cardinaux pour l'indexation des actions
        # 0: Nord, 1: Nord-Est, 2: Est, 3: Sud-Est, 4: Sud, 5: Sud-Ouest, 6: Ouest, 7: Nord-Ouest, 8: Rester en place
        
        # Préchargement de quelques comportements de base liés à la physique de la voile
        # Ces valeurs sont des heuristiques qui seront affinées par l'apprentissage
        
        # Règle générale: si le vent vient du nord, préférez les directions est/ouest pour louvoyer
        north_wind = (2, 2, 0, 0)  # Position près du départ, vent du nord
        self.q_table[north_wind] = np.array([0.1, 0.5, 0.9, 0.5, 0.1, 0.5, 0.9, 0.5, 0.0])
        
        # Si le vent vient du nord-est, louvoyez via le nord-ouest/sud-est
        northeast_wind = (2, 2, 0, 1)  # Position près du départ, vent du nord-est
        self.q_table[northeast_wind] = np.array([0.1, 0.1, 0.5, 0.9, 0.5, 0.1, 0.5, 0.9, 0.0])
        
        # Si le vent vient du nord-ouest, louvoyez via le nord-est/sud-ouest
        northwest_wind = (2, 2, 0, 7)  # Position près du départ, vent du nord-ouest
        self.q_table[northwest_wind] = np.array([0.1, 0.9, 0.5, 0.1, 0.5, 0.9, 0.5, 0.1, 0.0])
        
        # Pour les vents sud, préférez aller directement vers le nord (le plus efficace)
        south_wind = (2, 2, 0, 4)  # Position près du départ, vent du sud
        self.q_table[south_wind] = np.array([0.9, 0.7, 0.5, 0.3, 0.1, 0.3, 0.5, 0.7, 0.0])
        
    def discretize_state(self, observation):
        """Convertit l'observation continue en état discret pour la table Q."""
        # Extrait position, vitesse et vent de l'observation
        x, y = observation[0], observation[1]
        vx, vy = observation[2], observation[3]
        wx, wy = observation[4], observation[5]
        
        # Discrétisation de la position
        x_bin = min(int(x / self.grid_size[0] * self.position_bins), self.position_bins - 1)
        y_bin = min(int(y / self.grid_size[1] * self.position_bins), self.position_bins - 1)
        
        # Discrétisation de la direction de la vitesse (ignore la magnitude pour simplifier)
        v_magnitude = np.sqrt(vx**2 + vy**2)
        if v_magnitude < 0.1:  # Si la vitesse est très faible, bin spécial
            v_bin = 0
        else:
            v_direction = np.arctan2(vy, vx)  # Plage: [-pi, pi]
            v_bin = int(((v_direction + np.pi) / (2 * np.pi) * self.velocity_bins)) % self.velocity_bins
        
        # Discrétisation de la direction du vent
        wind_direction = np.arctan2(wy, wx)  # Plage: [-pi, pi]
        wind_bin = int(((wind_direction + np.pi) / (2 * np.pi) * self.wind_bins)) % self.wind_bins
        
        # Retourne le tuple d'état discret
        return (x_bin, y_bin, v_bin, wind_bin)
    
    def calculate_sailing_efficiency(self, boat_direction, wind_direction):
        """
        Calcule l'efficacité de navigation basée sur l'angle entre la direction du bateau et le vent.
        
        Args:
            boat_direction: Vecteur normalisé de la direction souhaitée du bateau
            wind_direction: Vecteur normalisé de la direction du vent (où le vent va VERS)
            
        Returns:
            sailing_efficiency: Flottant entre 0.05 et 1.0 représentant l'efficacité de navigation
        """
        # Inverser la direction du vent pour obtenir d'où vient le vent
        wind_from = -wind_direction
        
        # Calculer l'angle entre le vent et la direction
        cos_angle = np.dot(wind_from, boat_direction)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Éviter les erreurs numériques
        wind_angle = np.arccos(cos_angle)
        
        # Calcul de l'efficacité de navigation basé sur l'angle au vent
        if wind_angle < self.no_go_zone_angle:  # Moins de 30 degrés par rapport au vent
            sailing_efficiency = 0.05  # Efficacité faible mais non nulle dans la zone interdite
        elif wind_angle < self.close_hauled_angle:  # Entre 30 et 45 degrés
            sailing_efficiency = 0.1 + 0.4 * (wind_angle - self.no_go_zone_angle) / (self.close_hauled_angle - self.no_go_zone_angle)
        elif wind_angle < self.optimal_beam_reach_angle:  # Entre 45 et 90 degrés
            sailing_efficiency = 0.5 + 0.5 * (wind_angle - self.close_hauled_angle) / (self.optimal_beam_reach_angle - self.close_hauled_angle)
        elif wind_angle < 3*np.pi/4:  # Entre 90 et 135 degrés
            sailing_efficiency = 1.0  # Efficacité maximale
        else:  # Plus de 135 degrés
            sailing_efficiency = 1.0 - 0.2 * (wind_angle - 3*np.pi/4) / (np.pi/4)
            sailing_efficiency = max(0.8, sailing_efficiency)  # Mais toujours bien
        
        return sailing_efficiency
    
    def action_to_direction(self, action):
        """Convertit l'indice d'action en vecteur de direction."""
        directions = [
            (0, 1),     # 0: Nord
            (1, 1),     # 1: Nord-Est
            (1, 0),     # 2: Est
            (1, -1),    # 3: Sud-Est
            (0, -1),    # 4: Sud
            (-1, -1),   # 5: Sud-Ouest
            (-1, 0),    # 6: Ouest
            (-1, 1),    # 7: Nord-Ouest
            (0, 0)      # 8: Rester en place
        ]
        return np.array(directions[action])
    
    def get_upwind_tacking_action(self, position, wind_vector):
        """
        Implémente une stratégie de louvoyage pour naviguer contre le vent.
        Alterne entre les directions à environ 45 degrés de part et d'autre du vent.
        """
        # Normaliser le vecteur de vent
        wind_magnitude = np.linalg.norm(wind_vector)
        if wind_magnitude < 0.001:
            return 0  # Par défaut, aller au nord si pas de vent
            
        wind_normalized = wind_vector / wind_magnitude
        
        # Calcul de l'angle du vent (d'où il vient)
        wind_from_angle = np.arctan2(-wind_normalized[1], -wind_normalized[0])
        
        # Calculer les angles de louvoyage (environ 45 degrés de chaque côté du vent)
        port_tack_angle = wind_from_angle - self.close_hauled_angle
        starboard_tack_angle = wind_from_angle + self.close_hauled_angle
        
        # Décider de changer de bord si nécessaire
        distance_to_center = abs(position[0] - self.grid_size[0]/2)
        
        # Changer de bord si on s'écarte trop du centre ou si on est bloqué
        if distance_to_center > self.grid_size[0]/4 or self.steps_without_progress > 5:
            self.current_tack_direction *= -1
            self.steps_without_progress = 0
        
        # Sélectionner l'angle en fonction du bord actuel
        tack_angle = starboard_tack_angle if self.current_tack_direction > 0 else port_tack_angle
        
        # Convertir l'angle en vecteur de direction
        tack_direction = np.array([np.cos(tack_angle), np.sin(tack_angle)])
        
        # Trouver l'action la plus proche de cette direction
        best_action = 0
        best_similarity = -1
        
        for action in range(8):  # Exclure l'action "rester en place"
            action_direction = self.action_to_direction(action)
            # Calculer la similarité cosinus
            similarity = np.dot(tack_direction, action_direction) / (np.linalg.norm(tack_direction) * np.linalg.norm(action_direction))
            if similarity > best_similarity:
                best_similarity = similarity
                best_action = action
        
        return best_action
    
    def get_best_action_for_wind(self, position, wind_vector, goal_vector):
        """
        Détermine la meilleure action basée sur le vent, la position actuelle et l'objectif.
        Utilise les principes de la physique de la voile.
        """
        # Normaliser les vecteurs
        wind_magnitude = np.linalg.norm(wind_vector)
        if wind_magnitude < 0.001:
            # Si pas de vent, naviguer directement vers l'objectif
            return self.get_direct_goal_action(position, goal_vector)
            
        wind_normalized = wind_vector / wind_magnitude
        goal_direction = goal_vector / np.linalg.norm(goal_vector)
        
        # Déterminer si l'objectif est contre le vent
        # Le vent vient de l'opposé de wind_normalized
        wind_from_direction = -wind_normalized
        angle_to_goal = np.arccos(np.clip(np.dot(wind_from_direction, goal_direction), -1.0, 1.0))
        
        # Si l'objectif est dans la zone interdite (contre le vent), utiliser le louvoyage
        if angle_to_goal < self.close_hauled_angle:
            return self.get_upwind_tacking_action(position, wind_vector)
        
        # Sinon, choisir l'action qui offre la meilleure efficacité dans la direction de l'objectif
        best_action = 0
        best_efficiency = -1
        
        for action in range(8):  # Exclure l'action "rester en place"
            action_direction = self.action_to_direction(action)
            
            # Calculer l'efficacité de navigation pour cette direction
            sailing_efficiency = self.calculate_sailing_efficiency(
                action_direction / np.linalg.norm(action_direction), 
                wind_normalized
            )
            
            # Calculer combien cette action nous rapproche de l'objectif
            goal_alignment = np.dot(action_direction, goal_direction)
            
            # Score combiné: efficacité de navigation * alignement avec l'objectif
            combined_score = sailing_efficiency * (0.5 + 0.5 * goal_alignment)
            
            if combined_score > best_efficiency:
                best_efficiency = combined_score
                best_action = action
        
        return best_action
    
    def get_direct_goal_action(self, position, goal_vector):
        """Obtient l'action qui mène le plus directement vers l'objectif."""
        # Normaliser le vecteur d'objectif
        if np.linalg.norm(goal_vector) < 0.001:
            return 8  # Rester en place si on est déjà à l'objectif
            
        goal_direction = goal_vector / np.linalg.norm(goal_vector)
        
        # Trouver l'action la plus alignée avec la direction de l'objectif
        best_action = 0
        best_alignment = -1
        
        for action in range(8):  # Exclure l'action "rester en place"
            action_direction = self.action_to_direction(action)
            alignment = np.dot(action_direction, goal_direction) / np.linalg.norm(action_direction)
            
            if alignment > best_alignment:
                best_alignment = alignment
                best_action = action
        
        return best_action
    
    def analyze_wind_field(self, wind_field_flat, position):
        """
        Analyse le champ de vent complet pour identifier les zones favorables.
        Recherche des courants favorables ou des zones de vent fort dans la direction de l'objectif.
        """
        # Reconstruire le champ de vent 2D à partir des données aplaties
        wind_field = wind_field_flat.reshape(self.grid_size[1], self.grid_size[0], 2)
        
        # Calculer le vecteur vers l'objectif
        goal_vector = self.goal_position - position
        goal_distance = np.linalg.norm(goal_vector)
        if goal_distance < 0.001:
            return 8  # Déjà à l'objectif
            
        goal_direction = goal_vector / goal_distance
        
        # Rechercher des modèles favorables dans le champ de vent
        # (simpliste pour l'instant, pourrait être amélioré avec des algorithmes de clustering ou de recherche de chemin)
        
        # Vérifier s'il y a un couloir de vent favorable direct vers l'objectif
        favorable_path = True
        step_size = max(1, int(goal_distance / 5))  # Échantillonner 5 points sur le chemin
        
        for step in range(1, min(6, step_size + 1)):
            # Position intermédiaire sur la trajectoire
            intermediate_pos = position + goal_direction * step * (goal_distance / 5)
            x, y = int(intermediate_pos[0]), int(intermediate_pos[1])
            
            # Vérifier que la position est dans les limites
            if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
                # Obtenir le vent à cette position
                intermediate_wind = wind_field[y, x]
                
                # Calculer l'efficacité dans la direction de l'objectif
                efficiency = self.calculate_sailing_efficiency(
                    goal_direction,
                    intermediate_wind / (np.linalg.norm(intermediate_wind) + 1e-10)
                )
                
                # Si l'efficacité est faible à un point quelconque, le chemin n'est pas favorable
                if efficiency < 0.5:
                    favorable_path = False
                    break
        
        # Si un chemin favorable existe, naviguer directement vers l'objectif
        if favorable_path:
            return self.get_direct_goal_action(position, goal_vector)
        
        # Sinon, utiliser la stratégie basée sur le vent local
        return self.get_best_action_for_wind(position, wind_field[int(position[1]), int(position[0])], goal_vector)
    
    def act(self, observation):
        """
        Sélectionne une action basée sur l'observation actuelle.
        Utilise une combinaison de la table Q, de l'analyse du champ de vent et de stratégies de voile.
        """
        # Extraire les informations de l'observation
        position = np.array([observation[0], observation[1]])
        velocity = np.array([observation[2], observation[3]])
        wind_at_position = np.array([observation[4], observation[5]])
        wind_field_flat = observation[6:]
        
        # Vecteur vers l'objectif
        goal_vector = self.goal_position - position
        current_distance_to_goal = np.linalg.norm(goal_vector)
        
        # Suivre les progrès
        if self.previous_position is not None:
            distance_improvement = self.last_distance_to_goal - current_distance_to_goal
            if distance_improvement < 0.1:  # Si on ne progresse pas assez
                self.steps_without_progress += 1
            else:
                self.steps_without_progress = 0
                
        self.previous_position = position.copy()
        self.last_distance_to_goal = current_distance_to_goal
        
        # Exploration aléatoire avec faible probabilité (en phase de test)
        if self.np_random.random() < self.exploration_rate:
            return self.np_random.integers(0, 9)
        
        # Stratégie basée sur l'état de la table Q
        state = self.discretize_state(observation)
        if state in self.q_table:
            q_action = np.argmax(self.q_table[state])
            
            # Si nous sommes bloqués trop longtemps, ignorer la table Q
            if self.steps_without_progress < 10:
                return q_action
        
        # Si nous sommes ici, soit l'état n'est pas dans la table Q,
        # soit nous sommes bloqués et devons essayer une autre approche
        
        # Analyser le champ de vent si activé
        if self.use_wind_planning:
            wind_analysis_action = self.analyze_wind_field(wind_field_flat, position)
            return wind_analysis_action
            
        # Utiliser une approche basée sur la physique de la voile comme fallback
        return self.get_best_action_for_wind(position, wind_at_position, goal_vector)
    
    def reset(self):
        """Réinitialise l'agent pour un nouvel épisode."""
        self.previous_position = None
        self.steps_without_progress = 0
        self.last_distance_to_goal = float('inf')
        self.current_tack_direction = 1 if self.np_random.random() < 0.5 else -1
        
    def seed(self, seed=None):
        """Définit la graine aléatoire pour la reproductibilité."""
        self.np_random = np.random.default_rng(seed)