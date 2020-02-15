from preprocessing import Preprocesser
from imputation import Imputer
import os

MOVIESPATH = "/home/gonzalo/tfmunir/RecommendationEngine/RecommendationEngine/datos/tmdb_5000_movies.csv"
CREDITSPATH = "/home/gonzalo/tfmunir/RecommendationEngine/RecommendationEngine/datos/tmdb_5000_credits.csv"

PP = Preprocesser(movies_path = MOVIESPATH, credits_path= CREDITSPATH)
df = PP.preprocess()

IM = Imputer(df)

df = IM.impute()
print(df.head(5))