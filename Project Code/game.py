
VALID_GAMES = [
    "Alien", "Amidar", "Assault", "Asterix", "Asteroids", "Atlantis",
    "BankHeist", "BattleZone", "BeamRider", "Bowling", "Boxing", "Breakout",
    "Centipede", "ChopperCommand", "CrazyClimber", "DemonAttack", "DoubleDunk",
    "Enduro", "FishingDerby", "Freeway", "Frostbite", "Gopher", "Gravitar",
    "IceHockey", "Jamesbond", "Kangaroo", "Krull", "KungFuMaster",
    "MontezumaRevenge", "MsPacman", "NameThisGame", "Pitfall", "Pong",
    "PrivateEye", "Qbert", "Riverraid", "RoadRunner", "Robotank", "Seaquest",
    "SpaceInvaders", "StarGunner", "Tennis", "TimePilot", "Tutankham",
    "UpNDown", "Venture", "VideoPinball", "WizardOfWor", "Zaxxon",
] # Taken from the PPO paper directly

class Game:
    '''
    Game class, forces you to choose a game from the list.
    '''
    def __init__(self, name: str):
        if name not in VALID_GAMES:
            raise ValueError(f"{name} not a valid game. Check VALID_GAMES")
        self.name = name
        self.env_id = f"ALE/{self.name}-v5"

    