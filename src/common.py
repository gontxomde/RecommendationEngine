import pandas as pd
import json
from nltk.corpus import wordnet
import nltk

LOST_COLUMNS = [
    'actor_1_facebook_likes',
    'actor_2_facebook_likes',
    'actor_3_facebook_likes',
    'aspect_ratio',
    'cast_total_facebook_likes',
    'color',
    'content_rating',
    'director_facebook_likes',
    'facenumber_in_poster',
    'movie_facebook_likes',
    'movie_imdb_link',
    'num_critic_for_reviews',
    'num_user_for_reviews'
                ]

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

def load_tmdb_movies(path):
    """
    Función utilizada para cargar el dataset de las películas. Se transforma a fecha el campo de fecha de salida
    y se cargan como listas los campos que están guardados como json.

    Args:
        path: Ruta hasta el archivo de tmdb_5000_movies.csv

    Returns:
        Dataframe de pandas con la información del csv
    """
    df = pd.read_csv(path)
    df['release_date'] = pd.to_datetime(df['release_date']).apply(lambda x: x.date())
    json_columns = ['genres', 'keywords', 'production_countries', 'production_companies', 'spoken_languages']
    for column in json_columns:
        df[column] = df[column].apply(json.loads)
    return df


def load_tmdb_credits(path):
    """
    Función utilizada para cargar el dataset de los créditos. Se cargan como listas los campos que están guardado

    Args:
        path: Ruta hasta el archivo de tmdb_5000_credits.csv

    Returns:
        Dataframe de pandas con la información del csv
    """
    df = pd.read_csv(path)
    json_columns = ['cast', 'crew']
    for column in json_columns:
        df[column] = df[column].apply(json.loads)
    return df

def safe_access(container, index_values):
    """
    Función para acceder de forma segura a valores. En caso de que no se encuentre uno de ellos, se devuelve NaN
    en vez de lanzar un error.

    Args:
        container: Lista/ contenedor de la que quieren extraerse los valores
        index_values: Lista de índices a extraer del contenedor

    Returns:
        
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
    
    Arguments:
        crew_data -- JSON con el equipo que ha realizado la película
    
    Returns:
        str -- director de la película
    """
    directors = [x['name'] for x in crew_data if x['job'] == 'Director']
    return safe_access(directors, [0])


def pipe_flatten_names(keywords):
    """Obtiene una lista con las keywords separadas por un pipe | extrayéndolas del json.
    
    Arguments:
        keywords -- JSON de keywords de la película
    
    Returns:
        str -- keywords de la película juntas
    """
    return '|'.join([x['name'] for x in keywords])


def convert_to_original_format(movies, credits):
    """Aplica una serie de funciones para añadir información al dataset de películas a partir del
    conjunto de datos de créditos
    
    Arguments:
        movies -- DataFrame obtenido de leer el archivo de películas
        credits -- DataFrame obtenido de leet el archivo de créditos
    
    Returns:
        DataFrame -- Datos obtenidos de cruzar ambos conjuntos de datos.
    """
    tmdb_movies = movies.copy()
    tmdb_movies.rename(columns=TMDB_TO_IMDB_SIMPLE_EQUIVALENCIES, inplace=True)
    tmdb_movies['title_year'] = pd.to_datetime(tmdb_movies['release_date']).apply(lambda x: x.year)
    # I'm assuming that the first production country is equivalent, but have not been able to validate this
    tmdb_movies['country'] = tmdb_movies['production_countries'].apply(lambda x: safe_access(x, [0, 'name']))
    tmdb_movies['language'] = tmdb_movies['spoken_languages'].apply(lambda x: safe_access(x, [0, 'name']))
    tmdb_movies['director_name'] = credits['crew'].apply(get_director)
    tmdb_movies['actor_1_name'] = credits['cast'].apply(lambda x: safe_access(x, [1, 'name']))
    tmdb_movies['actor_2_name'] = credits['cast'].apply(lambda x: safe_access(x, [2, 'name']))
    tmdb_movies['actor_3_name'] = credits['cast'].apply(lambda x: safe_access(x, [3, 'name']))
    tmdb_movies['genres'] = tmdb_movies['genres'].apply(pipe_flatten_names)
    tmdb_movies['plot_keywords'] = tmdb_movies['plot_keywords'].apply(pipe_flatten_names)
    return tmdb_movies

def get_synonyms(keyword):
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

def keywords_inventory(dataframe, column = 'plot_keywords'):
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