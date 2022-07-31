import enum


class Model_Type(enum.Enum):
    pure_collaborative = "Pure Collaborative Model"
    colab_filtering = "Collaborative Filtering with GameNet Model"
    game_net = "GameNet Model"
