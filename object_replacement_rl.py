import os

from stable_baselines3 import SAC


class ObjectReplacementRL:

    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/model.zip")
        self.model = SAC.load(self.model_path)

    def get_action(self, distance: float, radius: float) -> float:
        return self.model.predict([[distance, radius]])[0][0][0]
