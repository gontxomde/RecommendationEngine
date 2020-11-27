import pandas as pd
import nltk

from RecommendationEngine.common import load_credits
from RecommendationEngine.common import load_movies
from RecommendationEngine.common import combine_collections
from RecommendationEngine.common import keywords_inventory
from RecommendationEngine.common import get_synonyms


class Preprocessor:
    def __init__(self, movies_path, credits_path):

        credit = load_credits(credits_path)
        movies = load_movies(movies_path)
        self.df = combine_collections(movies, credit)
        self.keywords_inventory = keywords_inventory
        self.get_synonyms = get_synonyms

        # Variables needed

        self.new_col_order = ['movie_title', 'title_year', 'genres', 'plot_keywords',
                              'director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name',
                              'num_voted_users', 'language', 'country', 'vote_average', 'duration',
                              'gross']

    def preprocess(self):
        self.set_keywords = set()
        for list_keywords in self.df['plot_keywords'].str.split('|').values:
            if isinstance(list_keywords, float): continue  # Evitar las películas en las que no hay keywords
            self.set_keywords = self.set_keywords.union(list_keywords)
        # Keywords y número de apariciones
        self.keyword_occurences, _ = self.count_word(self.df, 'plot_keywords', self.set_keywords)

        # Se retiran las keywords que son vacías
        self.keyword_occurences = [x for x in self.keyword_occurences if x[0]]
        _, _, keywords_select = self.keywords_inventory(self.df, column='plot_keywords')
        self.df = self.df_keywords_replacement(self.df, keywords_select, roots=True)

        self.keyword_occurences.sort(key=lambda x: x[1], reverse=False)
        key_count = dict()
        for s in self.keyword_occurences:
            key_count[s[0]] = s[1]
        # __________________________________________________________________________
        # Creación de un diccionario para reemplazar keywords por sinónimos de mayor frecuencia
        remplacement_word = dict()
        icount = 0
        for _, [word, nb_apparitions] in enumerate(self.keyword_occurences):
            if nb_apparitions > 5: continue  # Sólo las keywords que aparecen menos de 5 veces
            lemma = self.get_synonyms(word)
            if len(lemma) == 0: continue  # Caso de plurales
            # _________________________________________________________________
            word_list = [(s, key_count[s]) for s in lemma
                         if self.test_keyword(s, key_count, key_count[word])]
            word_list.sort(key=lambda x: (x[1], x[0]), reverse=True)
            if len(word_list) <= 1: continue  # NO se reemplaza
            if word == word_list[0][0]: continue  # Reemplazo por sí mismo
            icount += 1
        # Keywords that appear both in keys and values:
        icount = 0
        for s in remplacement_word.values():
            if s in remplacement_word.keys():
                icount += 1
                if icount < 10: print('{:<20} -> {:<20}'.format(s, remplacement_word[s]))

        for key, value in remplacement_word.items():
            if value in remplacement_word.keys():
                remplacement_word[key] = remplacement_word[value]

        # Se reemplazan variaciones de una keyword por su keyword principal
        # ----------------------------------------------------------
        self.df = \
            self.df_keywords_replacement(self.df, remplacement_word, roots=False)
        _, _, keywords_select = \
            self.keywords_inventory(self.df, column='plot_keywords')

        self.df = self.df[self.new_col_order]

        return self.df

    def count_word(self, df, ref_col, lista):
        """Toma una columna de un dataframe y un set de valores y devuelve un diccionario y una lista
        con el numero de apariciones de cada elemento de la lista en la columna del dataframe
        
        Args:
            df (pd.DataFrame]): DataFrame del que extraer la información
            ref_col (str): Columna de la que extraer los valores diferentes
            lista (list): Lista con los diferentes valores de los que extraer sus apariciones
        Returns:
            list: Número de apariciones
            dict: Número de apariciones
            
        """
        keyword_count = dict()
        for s in lista: keyword_count[s] = 0
        for lista_keywords in df[ref_col].str.split('|'):
            if type(lista_keywords) == float and pd.isnull(lista_keywords): continue
            # for s in lista:
            for s in [s for s in lista_keywords if s in lista]:
                if pd.notnull(s): keyword_count[s] += 1
        # ______________________________________________________________________
        # convert the dictionary in a list to sort the keywords by frequency
        keyword_occurences = []
        for k, v in keyword_count.items():
            keyword_occurences.append([k, v])
        keyword_occurences.sort(key=lambda x: x[1], reverse=True)
        return keyword_occurences, keyword_count

    def df_keywords_replacement(self, df, replacement_dict, roots=False, column='plot_keywords'):
        """Reemplaza las palabras clave de una película por las formas básicas de las mismas.
        
        Args:
            df (pd.DataFrame): DataFrame que contiene la información de las películas
            replacement_dict (dict): Diccionario con los cambios
            roots (bool, optional): Controla si se obtienen las raices de las palabras de las
            keywords. Defaults to False.
            column (str, optional): Columna en la que realizar la transformación. Defaults to 'plot_keywords'.
        
        Returns:
            pd.DataFrame: DataFrame con las sustituciones realizadas
        """

        PS = nltk.stem.PorterStemmer()
        df_new = df.copy(deep=True)
        for index, row in df_new.iterrows():
            chain = row[column]
            if pd.isnull(chain): continue
            new_list = []
            for s in chain.split('|'):
                key = PS.stem(s) if roots else s
                if key in replacement_dict.keys():
                    new_list.append(replacement_dict[key])
                else:
                    new_list.append(s)
            df_new.at[index, column] = '|'.join(new_list)
        return df_new

    def test_keyword(self, word, key_count, threshold):
        """Devuelve si una palabra aparece un número mayor de veces que el umbral señalado
        
        Args:
            word (str): Palabra a buscar
            key_count (dict): Diccionario con las apariciones de cada keyword
            threshold (int): Umbral
        
        Returns:
            bool: True si aparece un número mayor de veces
        """
        return (False, True)[key_count.get(word, 0) >= threshold]

    def replacement_df_low_frequency_keywords(self, df, keyword_occurences):
        """Modifica las entradas del dataframe, quitando las keywords que aparecen menos 
        de 3 veces.
        
        Args:
            df (pd.DataFrame): DataFrame de películas
            keyword_occurences ([type]): Diccionario que contiene la ocurrencia de cada keyword
        
        Returns:
            pd.Dataframe: DataFrame con las nuevas keywords
        """
        df_new = df.copy(deep=True)
        key_count = dict()
        for s in keyword_occurences:
            key_count[s[0]] = s[1]
        for index, row in df_new.iterrows():
            chain = row['plot_keywords']
            if pd.isnull(chain): continue
            new_list = []
            for s in chain.split('|'):
                if key_count.get(s, 4) > 3: new_list.append(s)
            df_new.at[index, 'plot_keywords'] = '|'.join(new_list)
        return df_new
