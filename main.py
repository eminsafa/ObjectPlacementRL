from stable_baselines3 import SAC


def get_action(distance: float, radius: float) -> float:
    return SAC.load('models/model.zip').predict([[distance, radius]])[0][0][0]
