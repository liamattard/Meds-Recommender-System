import enum


class Model_Type(enum.Enum):
    #Deprecated
    game_net_age = "GameNet Model with age"
    game_net_item_coll = "GameNet w Item based Collaborative Model"
    game_net_age_item_coll = "GameNet w Item based Collaborative Model with age"
    top_20 = "Model that recommends the top 20 most popular medicine"
    pure_collaborative = "Pure Collaborative Model"
    colab_filtering = "Collaborative Filtering with GameNet Model"

    # Final
    final_model = "Demographic and Collaborative Filtering"
    game_net = "GameNet Model"
    game_net_coll = "GameNet w Collaborative Filtering"
    game_net_knn = "Demographic Based GameNet w KNN"