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

from common import load_tmdb_credits
from common import load_tmdb_movies
from common import convert_to_original_format

class Preprocesser():
    def __init__(self,movies_path, credits_path):

        credit = load_tmdb_credits(credits_path)
        movies = load_tmdb_movies(movies_path)
        self.df = convert_to_original_format(movies, credit)



        #Variables needed

        self.new_col_order = ['movie_title', 'title_year', 'genres', 'plot_keywords',
         'director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name', 
         'num_voted_users', 'language', 'country', 'vote_average', 'duration', 
         'gross']

    def preprocess(self):
        self.set_keywords = set()
        for list_keywords in self.df['plot_keywords'].str.split('|').values:
            if isinstance(list_keywords, float): continue  # Evitar las películas en las que no hay keywords
            self.set_keywords = self.set_keywords.union(list_keywords)
        #Keywords y número de apariciones
        self.keyword_occurences, _ = self.count_word(self.df, 'plot_keywords', self.set_keywords)

        # Se retiran las keywords que son vacías
        self.keyword_occurences = [x for x in self.keyword_occurences if x[0]]
        _, _, keywords_select = self.keywords_inventory(self.df, column = 'plot_keywords')
        self.df = self.df_keywords_replacement(self.df, keywords_select, roots = True)

        self.keyword_occurences.sort(key = lambda x:x[1], reverse = False)
        key_count = dict()
        for s in self.keyword_occurences:
            key_count[s[0]] = s[1]
        #__________________________________________________________________________
        # Creación de un diccionario para reemplazar keywords por sinónimos de mayor frecuencia
        remplacement_word = dict()
        icount = 0
        for _, [word, nb_apparitions] in enumerate(self.keyword_occurences):
            if nb_apparitions > 5: continue  # Sólo las keywords que aparecen menos de 5 veces
            lemma = self.get_synonyms(word)
            if len(lemma) == 0: continue     #Caso de plurales
            #_________________________________________________________________
            word_list = [(s, key_count[s]) for s in lemma 
                        if self.test_keyword(s, key_count, key_count[word])]
            word_list.sort(key = lambda x:(x[1],x[0]), reverse = True)    
            if len(word_list) <= 1: continue       # NO se reemplaza
            if word == word_list[0][0]: continue    # Reemplazo por sí mismo
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
        #----------------------------------------------------------
        self.df = \
                    self.df_keywords_replacement(self.df, remplacement_word, roots = False)   
        _, _, keywords_select = \
                    self.keywords_inventory(self.df, column = 'plot_keywords')

        self.df = self.df[self.new_col_order]

        return self.df


    def count_word(self, df, ref_col, lista):
        """Toma una columna de un dataframe y un set de valores y devuelve un diccionario y una lista
        con el numero de apariciones de cada elemento de la lista en la columna del dataframe
        
        Arguments:
            df -- DataFrame del que extraer la información
            ref_col -- Columna de la que extraer los valores diferentes
            lista -- Lista con los diferentes valores de los que extraer sus apariciones
        
        Returns:
            lista y diccionario con el número de apariciones
        """
        keyword_count = dict()
        for s in lista: keyword_count[s] = 0
        for lista_keywords in df[ref_col].str.split('|'):
            if type(lista_keywords) == float and pd.isnull(lista_keywords): continue
            #for s in lista:
            for s in [s for s in lista_keywords if s in lista]:
                if pd.notnull(s): keyword_count[s] += 1
        #______________________________________________________________________
        # convert the dictionary in a list to sort the keywords by frequency
        keyword_occurences = []
        for k,v in keyword_count.items():
            keyword_occurences.append([k,v])
        keyword_occurences.sort(key = lambda x:x[1], reverse = True)
        return keyword_occurences, keyword_count
    
    def keywords_inventory(self, dataframe, column = 'plot_keywords'):
        """Devuelve un diccionario con las palabras que derivan de cada lexema
        a partir de un DataFrame y la columna de la que se quiere extraer
        
        Arguments:
            dataframe -- DataFrame del que obtener la información
        
        Keyword Arguments:
            column -- Nombre de la columna (default: {'plot_keywords'})
        
        Returns:
            Lista con las keywords finales que aparecen
            Diccionario con la relación lexema <-> palabras
            Diccionario con la palabra más corta derivada del lexema
        """
        PS = nltk.stem.PorterStemmer()
        keywords_roots  = dict()  # recoger las palabras de cada lexema
        keywords_select = dict()  # asociacion: lexema <-> keyword
        category_keys = []
        for s in dataframe[column]:
            if pd.isnull(s): continue
            for t in s.split('|'):
                t = t.lower() ; root = PS.stem(t)
                # Para cada lexema, un set con las palabras que lo usan
                if root in keywords_roots:                
                    keywords_roots[root].add(t)
                else:
                    keywords_roots[root] = {t}
        
        for s in keywords_roots.keys():
            if len(keywords_roots[s]) > 1:  
                min_length = 1000
                for k in keywords_roots[s]:
                    if len(k) < min_length:
                        key = k ; min_length = len(k)            
                category_keys.append(key)
                keywords_select[s] = key
            else:
                category_keys.append(list(keywords_roots[s])[0])
                keywords_select[s] = list(keywords_roots[s])[0]
                    
        #print("Número de keywords en la variable: '{}': {}".format(column,len(category_keys)))
        return category_keys, keywords_roots, keywords_select

    def df_keywords_replacement(self, df, replacement_dict, roots = False, column = 'plot_keywords'):
        """Reemplaza las palabras clave de una película por las formas básicas de las mismas.
        
        Arguments:
            df -- DataFrame que contiene la información de las películas
            replacement_dict -- diccionarion]
        
        Keyword Arguments:
            roots {bool} -- Controla si se obtienen las raices de las palabras de las
            keywords (default: {False})
            column -- Columna en la que realizar la transformación
        
        Returns:
            df_new -- DataFrame con las sustituciones realizadas
        """
        PS = nltk.stem.PorterStemmer()
        df_new = df.copy(deep = True)
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
    
    def get_synonyms(self, keyword):
        """Se obtienen los sinónimos sustantivos de una palabra
        
        Arguments:
            keyword -- Palabra de la que obtener los sinónimos


        Returns:
            lemma -- Lista con los sinónimos
        """
    
        lemma = set()
        for ss in wordnet.synsets(keyword):
            for w in ss.lemma_names():
                #_______________________________
                #  Obtenemos los sinónimos que son sustantivos
                index = ss.name().find('.')+1
                if ss.name()[index] == 'n': lemma.add(w.lower().replace('_',' '))
        return lemma

    def test_keyword(self,word, key_count, threshold):
        """Devuelve si una palabra aparece un número mayor de veces que el umbral señalado
        
        Arguments:
            word -- Palabra a busvcar
            key_count -- Diccionario con las apariciones de cada keyword
            threshold -- Umbral
        
        Returns:
            bool -- True si aparece un número mayor de veces
        """
        return (False , True)[key_count.get(word, 0) >= threshold]

    def replacement_df_low_frequency_keywords(self, df, keyword_occurences):
        """Modifica las entradas del dataframe, quitando las keywords que aparecen menos de 3 veces.
        
        Arguments:
            df -- DataFrame de películas
            keyword_occurences -- Diccionario que contiene la ocurrencia de cada keyword
        
        Returns:
            df -- DataFrame con las nuevas keywords
        """
        df_new = df.copy(deep = True)
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