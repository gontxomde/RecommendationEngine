import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import math, nltk, warnings
from nltk.corpus import wordnet
from sklearn import linear_model
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz
from wordcloud import WordCloud, STOPWORDS

class Recommendator ():
    def __init__(self, df):
        self.df = df.copy(deep = True)
        self.gaussian_filter = lambda x,y,sigma: math.exp(-(x-y)**2/(2*sigma**2))
    
    def entry_variables(self, df, id_entry): 
        """Calcula los valores tomados por las variables director_name, actor_[1,2,3]_name y plot_keywords para la
        película seleccionada por el usuario.
        
        Args:
            df (pd.DataFrame): DataFrame de películas
            id_entry (int): Id de la entrada seleccionada
        
        Returns:
            list: Lista que contiene los valores extraidos para la película seleccionada
        """

        col_labels = []    
        if pd.notnull(df['director_name'].iloc[id_entry]):
            for s in df['director_name'].iloc[id_entry].split('|'):
                col_labels.append(s)
                
        for i in range(3):
            column = 'actor_NUM_name'.replace('NUM', str(i+1))
            if pd.notnull(df[column].iloc[id_entry]):
                for s in df[column].iloc[id_entry].split('|'):
                    col_labels.append(s)
                    
        if pd.notnull(df['plot_keywords'].iloc[id_entry]):
            for s in df['plot_keywords'].iloc[id_entry].split('|'):
                col_labels.append(s)
        return col_labels

    def add_variables(self, df, REF_VAR):
        """Añade al dataframe de películas las columnas dadas en REF_VAR (que serán el director, etc de una
        película) y las inicializa a 0 o 1 dependiendo de si la película es del mismo director, tiene a ese actor
        , etc
        
        Args:
            df (pd.DataFrame): DataFrame de películas
            REF_VAR (list): Salida de aplicar entry_variables sobre el df y una película
        
        Returns:
            pd.DataFrame: DataFrame con las nuevas películas
        """
        for s in REF_VAR: 
            df[s] = pd.Series([0 for _ in range(len(df))])
        columns = ['genres', 'actor_1_name', 'actor_2_name',
                    'actor_3_name', 'director_name', 'plot_keywords']
        for category in columns:
            for index, row in df.iterrows():
                if pd.isnull(row[category]): 
                    continue
                for s in row[category].split('|'):
                    if s in REF_VAR: df.at[index, s] = 1
        return df

    def recommend(self,df, id_entry, N = 31):
        """Crea una lista de N películas similares a las seleccionadas por el usuario
        
        Args:
            df (pd.DataFrame): DataFrame de películas
            id_entry (int): Id de la entrada seleccionada
            N (int, optional): Number of films recommended 
            (take into account that the nearest will be always itself). Defaults to 31.

        Returns:
            list: List of ids of films recommended
        """
        df_copy = df.copy(deep = True)    
        list_genres = set()
        for s in df['genres'].str.split('|').values:
            list_genres = list_genres.union(set(s))    
        #_____________________________________________________
        # Creación de variables adicionales para comprobar la similaridad
        variables = self.entry_variables(df_copy, id_entry)
        variables += list(list_genres)
        df_new = self.add_variables(df_copy, variables)
        #____________________________________________________________________________________
        # Determinación de los vecinos más próximos: la distancia se calcula con las nuevas vairables
        X = df_new[variables].values
        nbrs = NearestNeighbors(n_neighbors=N, algorithm='auto', metric='euclidean').fit(X)

        _, indices = nbrs.kneighbors(X)    
        xtest = df_new.iloc[id_entry][variables].values
        xtest = xtest.reshape(1, -1)

        _, indices = nbrs.kneighbors(xtest)

        return indices[0][:]

    def extract_parameters(self,df, list_films, N = 31):
        """Extrae algunas variables del dataframe dado como entrada y devuelve la lista de N películas.
        Esta lista se ordena de acuerdo al criterio de la función selection_criteria.
        
        Args:
            df ([type]): DataFrame de películas
            list_films (list): Lista con las n películas recomendadas
            N (int, optional): Number of films recommended. Defaults to 31.
        
        Returns:
            list: Películas recomendadas
        """
        parametres_films = ['_' for _ in range(N)]
        i = 0
        max_users = -1
        for index in list_films:
            parametres_films[i] = list(df.iloc[index][['movie_title', 'title_year',
                                            'vote_average', 
                                            'num_voted_users']])
            parametres_films[i].append(index)
            max_users = max(max_users, parametres_films[i][4] )
            i += 1
        # The first element is the selected film itself
        title_main = parametres_films[0][0]
        ref_year  = parametres_films[0][1]
        parametres_films.sort(key = lambda x:self.selection_criteria(title_main, max_users,
                                                                ref_year, 
                                                                title = x[0], 
                                                                year = x[1],
                                                                score = x[2], 
                                                                votes = x[3]), reverse = True)
        
        return parametres_films

    def sequel(self, title_1, title_2):   
        """Compara los títulos de dos películas y devuelve si son similares o no
        
        Args:
            title_1 (str): Primer título
            title_2 (str): Segundo título
        
        Returns:
            bool: True if the films are sequels. False otherwise.
        """
        #print("$$$$$$$$$$$$$$$$$$$$$$")
        #print(title_1, "|",title_2)
        #print(fuzz.ratio(title_1, title_2) , fuzz.token_set_ratio(title_1, title_2))
        if fuzz.ratio(title_1, title_2) > 50 or fuzz.token_set_ratio(title_1, title_2) > 60:
            return True
        else:
            return False

    def selection_criteria(self, title_main, max_users, ref_year, title, year, score, votes):
        """Calcula la puntuación de una película como recomendación de otra en base a la similaridad
        de su título, la distancia temporal entre ambos lanzamientos y el número de votos de la película evaluada
        y la puntuación de la película en IMDB.
        Además, la similitud entre títulos se tiene en cuenta para evitar la recomendación de secuelas. Es decir, 
        si dos películas tienen un nombre muy similar, se desechara como recomendación.
        
        Args:
            title_main (str): Título de la película dada por el usuario
            max_users (int): Máximo número de votos de las N películas
            ref_year (int): Año de lanzamiento de la película dada por el usuario
            title (str): Título de la película a evaluar
            year (int): Año de lanzamiento de la película a evaluar
            score (float): Votación media de la película a evaluar
            votes (int): Votos de la película a evaluar
        
        Returns:
            float: Mark of the film given
        """
        if pd.notnull(ref_year):
            factor_1 = self.gaussian_filter(ref_year, year, 20)
        else:
            factor_1 = 1        

        sigma = max_users * 1.0

        if pd.notnull(votes):
            factor_2 = self.gaussian_filter(votes, max_users, sigma)
        else:
            factor_2 = 0
            
        if self.sequel(title_main, title):
            mark = 0
            #print(f"Tenemos sequel entre {title_main} y {title}")
        else:
            mark = score * factor_1 * factor_2
        #print(f"'La nota de {title} es: {mark}'")
        return mark

    def add_to_selection(self, film_selection, parameters_films, N = 31, M = 5):
        """Completa la lista film_selection que contiene 5 películas que se recomendarán al usuario. Las películas
        son seleccionadas de parameters_list y sólo se tienen en cuenta si el título es suficientemente
        distinto del de otras películas.
        
        Args:
            film_selection (list): Lista de películas
            parameters_films (list): Lista de parámetros
            N (int, optional): Películas a puntuar. Defaults to 31.
            M (int, optional): Películas a recomendar. Defaults to 5.
        
        Returns:
            list: films reselected
        """
        film_list = film_selection[:]
        icount = len(film_list)    
        for i in range(N):
            already_in_list = False
            for s in film_selection:
                if s[0] == parameters_films[i][0]: 
                    already_in_list = True
                if self.sequel(parameters_films[i][0], s[0]): 
                    already_in_list = True            
            if already_in_list: continue
                
            icount += 1
            if icount <= M:
                film_list.append(parameters_films[i])
        return film_list

    def remove_sequels(self, film_selection):
        """Removes sequels from the list of films given
        
        Args:
            film_selection (list): Lista de películas de la que quitar las secuelas
        
        Returns:
            list: Lista sin secuelas
        """ 
        removed_from_selection = []
        for i, film_1 in enumerate(film_selection):
            for j, film_2 in enumerate(film_selection):
                if j <= i: continue 
                if self.sequel(film_1[0], film_2[0]): 
                    last_film = film_2[0] if film_1[1] < film_2[1] else film_1[0]
                    removed_from_selection.append(last_film)

        film_list = [film for film in film_selection if film[0] not in removed_from_selection]

        return film_list

    def find_similarities(self, df, id_entry, del_sequels = True, N = 31, M = 5):
        """Dado el id de una película busca las 5 mejores recomendaciones.
        
        Args:
            df (pd.DataFrame): [description]
            id_entry (int): [description]
            del_sequels (bool, optional): Borrar secuelas de las recomendaciones. Defaults to True.
            N (int, optional): Películas a evaluar. Defaults to 31.
            M (int, optional): Películas a recomendar. Defaults to 31.

        Returns:
            list: Selección de películas recomendadas
        """
        #____________________________________
        list_films = self.recommend(df, id_entry, N)
        #__________________________________
        # Crear lista de N películas
        parameters_films = self.extract_parameters(df, list_films, N)
        #print("&&\n",parameters_films)
        #_______________________________________
        # Seleccionar 5 películas de la  listaSelect 5 films from this list
        film_selection = []
        film_selection = self.add_to_selection(film_selection, parameters_films, N, M)
        #print("&&\n",film_selection)
        #__________________________________
        # Borrado de las secuelas
        if del_sequels: film_selection = self.remove_sequels(film_selection)
        #______________________________________________
        # Añadir nuevas películas a la lista
        #print(film_selection)
        film_selection = self.add_to_selection(film_selection, parameters_films, N, M)
        #_____________________________________________
        selection_titles = []
        for _,s in enumerate(film_selection):
            selection_titles.append([s[0].replace(u'\xa0', u''), s[4]])
            #if verbose: print("nº{:<2}     -> {:<30}".format(i+1, s[0]))

        return selection_titles
    
    @staticmethod
    def fuzzing(series_element, string):
        return fuzz.token_set_ratio(series_element, string)

    def string_to_id(self, df, string):
        df_2 = df.copy(deep = True)
        df_2["mark"] = df_2["movie_title"].apply(self.fuzzing , args = (string,))
        df_2.sort_values(by = ["mark", "title_year"], ascending = [False, True], inplace = True)
        return df_2.index.values[0]
    
    def predict_from_string(self, string):
        df = self.df.copy(deep = True)
        id_entry = self.string_to_id(df, string)

        selection_titles = self.find_similarities(df, id_entry, del_sequels = True, N = 31, M = 5)

        print("Para la película", df.loc[id_entry, "movie_title"], "te recomendamos ver:")
        for element in selection_titles:
            print("\t -", element[0])
