from preprocessing import Preprocesser
from imputation import Imputer

PP = Preprocesser(movies_path = "./datos/tmdb_5000_movies.csv", credits_path= "./datos/tmdb_5000_credits.csv")
df = PP.preprocess()

IM = Imputer(df)

df = IM.impute()
print(df.head(5))