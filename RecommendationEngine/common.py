import pandas as pd
import json
from nltk.corpus import wordnet
import nltk

TMDB_TO_IMDB_SIMPLE_EQUIVALENCIES = {
    'budget': 'budget',
    'genres': 'genres',
    'revenue': 'gross',
    'title': 'movie_title',
    'runtime': 'duration',
    'original_language': 'language',  # it's possible that spoken_languages would be a better match
    'keywords': 'plot_keywords',
    'vote_count': 'num_voted_users',
                                         }

IMDB_COLUMNS_TO_REMAP = {'imdb_score': 'vote_average'}


def load_movies(path):
    """Función utilizada para cargar el dataset de las películas. Se transforma a fecha el campo de fecha de salida
    y se cargan como listas los campos que están guardados como json.
    
    Args:
        path (str): Ruta hasta el archivo de tmdb_5000_movies.csv
    
    Returns:
        pd.DataFrame: Dataframe de pandas con la información del csv
    """

    df = pd.read_csv(path)
    df['release_date'] = pd.to_datetime(df['release_date']).apply(lambda x: x.date())
    json_columns = ['genres', 'keywords', 'production_countries', 'production_companies', 'spoken_languages']
    for column in json_columns:
        df[column] = df[column].apply(json.loads)
    return df

def load_credits(path):
    """Función utilizada para cargar el dataset de los créditos. Se cargan como listas los campos que están guardado
    
    Args:
        path (str): Ruta hasta el archivo de tmdb_5000_credits.csv

    
    Returns:
        pd.DataFrame: [description]
    """

    df = pd.read_csv(path)
    json_columns = ['cast', 'crew']
    for column in json_columns:
        df[column] = df[column].apply(json.loads)
    return df

def get_element(container, index_values):
    """Función para acceder de forma segura a valores. En caso de que no se encuentre uno de ellos, se devuelve NaN
    en vez de lanzar un error.
    
    Args:
        container ([type]): Lista/ contenedor de la que quieren extraerse los valores
        index_values ([type]): Lista de índices a extraer del contenedor
    
    Returns:
        any: Valores extraidos
    """

    result = container
    try:
        for idx in index_values:
            result = result[idx]
        return result
    except (IndexError, KeyError):
        return pd.np.nan

def get_director(crew_data):
    """Devuelve el director dado un json con toda la composición del equipo de la película.
    
    Args:
        crew_data (json): JSON con el equipo que ha realizado la película
    
    Returns:
        str: Director de la película
    """

    directors = [x['name'] for x in crew_data if x['job'] == 'Director']
    return get_element(directors, [0])

def pipe_flatten_names(keywords):
    """Obtiene una lista con las keywords separadas por un pipe | extrayéndolas del json.
    
    Args:
        keywords (json): keywords de la película
    
    Returns:
        str: keywords de la película juntas
    """
    return '|'.join([x['name'] for x in keywords])

def combine_collections(movies, credits):
    """Aplica una serie de funciones para añadir información al dataset de películas a partir del
    conjunto de datos de créditos
    
    Args:
        movies (pd.DataFrame): DataFrame obtenido de leer el archivo de películas
        credits (pd.DataFrame): DataFrame obtenido de leet el archivo de créditos
    
    Returns:
        pd.DataFrame: DataFrame con la información conjunta
    """

    tmdb_movies = movies.copy()
    tmdb_movies.rename(columns=TMDB_TO_IMDB_SIMPLE_EQUIVALENCIES, inplace=True)
    tmdb_movies['title_year'] = pd.to_datetime(tmdb_movies['release_date']).apply(lambda x: x.year)
    # I'm assuming that the first production country is equivalent, but have not been able to validate this
    tmdb_movies['country'] = tmdb_movies['production_countries'].apply(lambda x: get_element(x, [0, 'name']))
    tmdb_movies['language'] = tmdb_movies['spoken_languages'].apply(lambda x: get_element(x, [0, 'name']))
    tmdb_movies['director_name'] = credits['crew'].apply(get_director)
    tmdb_movies['actor_1_name'] = credits['cast'].apply(lambda x: get_element(x, [1, 'name']))
    tmdb_movies['actor_2_name'] = credits['cast'].apply(lambda x: get_element(x, [2, 'name']))
    tmdb_movies['actor_3_name'] = credits['cast'].apply(lambda x: get_element(x, [3, 'name']))
    tmdb_movies['genres'] = tmdb_movies['genres'].apply(pipe_flatten_names)
    tmdb_movies['plot_keywords'] = tmdb_movies['plot_keywords'].apply(pipe_flatten_names)
    return tmdb_movies

def get_synonyms(keyword):
    """Se obtienen los sinónimos sustantivos de una palabra
    
    Args:
        keyword (str): Palabra de la que obtener los sinónimos
    
    Returns:
        list: Lista con los sinónimos
    """

    lemma = set()
    for ss in wordnet.synsets(keyword):
        for w in ss.lemma_names():
            #_______________________________
            #  Obtenemos los sinónimos que son sustantivos
            index = ss.name().find('.')+1
            if ss.name()[index] == 'n': lemma.add(w.lower().replace('_',' '))
    return lemma

def keywords_inventory(dataframe, column = 'plot_keywords'):
    """Devuelve un diccionario con las palabras que derivan de cada lexema
    a partir de un DataFrame y la columna de la que se quiere extraer
    
    Args:
        dataframe (pd.DataFrame): DataFrame del que obtener la información.
        column (str, optional): Nombre de la columna. Defaults to 'plot_keywords'.
    
    Returns:
        list: Keywords finales que aparecen
        dict: Relación lexema <-> palabras
        dict: Palabra más corta derivada del lexema
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