import enum


class Model_Type(enum.Enum):
    game_net_age = "GameNet Model with age"
    game_net = "GameNet Model"
    game_net_coll = "GameNet w Collaborative Model"
    game_net_item_coll = "GameNet w Item based Collaborative Model"

    #Deprecated
    pure_collaborative = "Pure Collaborative Model"
    colab_filtering = "Collaborative Filtering with GameNet Model"
