
import pickle
import numpy as np
import os

def load_trained_sailing_model(agent, 
                               model_path="/home/onyxia/work/RL_project_sailing/notebooks/models/wind_aware_navigator_trained.pkl"):
    """
    Charge les poids entraînés dans l'agent WindAwareNavigator.

    Args:
        agent: L'agent WindAwareNavigator
        model_path: Chemin vers le fichier de poids (par défaut: celui existant)

    Returns:
        success: True si le chargement a réussi, False sinon
    """
    try:
        if not os.path.exists(model_path):
            print(f"Fichier de modèle non trouvé: {model_path}")
            return False

        with open(model_path, 'rb') as f:
            models = pickle.load(f)

            if 'state_eval_model' in models and 'wind_prediction_model' in models:
                agent.state_eval_model = models['state_eval_model']
                agent.wind_prediction_model = models['wind_prediction_model']
                print(f"Modèle chargé avec succès depuis {model_path}")
                return True
            else:
                print("Format de modèle invalide - clés manquantes")
                return False

    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        return False
