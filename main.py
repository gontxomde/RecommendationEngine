from preprocessing import Preprocesser

PP = Preprocesser(movies_path = "./datos/tmdb_5000_movies.csv", credits_path= "./datos/tmdb_5000_credits.csv")
df = PP.preprocess()
print(df.head(5))