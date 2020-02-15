from preprocessing import Preprocesser
from cleaning import Cleaner

PP = Preprocesser(movies_path = "./datos/tmdb_5000_movies.csv", credits_path= "./datos/tmdb_5000_credits.csv")
df = PP.preprocess()

CL = Cleaner(df)

df = CL.clean_df()
print(df.head(5))