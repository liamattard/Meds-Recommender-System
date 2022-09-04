import enum


class Model_Type(enum.Enum):
    pure_collaborative = "Pure Collaborative Model"
    colab_filtering = "Collaborative Filtering with GameNet Model"
    game_net_age = "GameNet Model with age"
    game_net = "GameNet Model"
