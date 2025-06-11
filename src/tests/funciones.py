import re
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def limpiar_tokens(text):
    """
    Limpia texto eliminando URLs, menciones, hashtags, números y signos de puntuación.
    Devuelve una lista de tokens en minúscula sin stopwords en español.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"@\w+|#|\d+", '', text)
    text = re.sub(r"[^\w\s]", '', text)
    tokens = word_tokenize(text)
    return [word for word in tokens if word not in stopwords.words('spanish') and len(word) > 2]

def limpiar_texto(text):
    """
    Devuelve un string limpio, juntando los tokens separados por espacios.
    """
    tokens = limpiar_tokens(text)
    return ' '.join(tokens)

def vector_promedio(tokens, modelo):
    """
    Genera el vector promedio de una lista de tokens usando un modelo Word2Vec.
    """
    vectores = [modelo.wv[token] for token in tokens if token in modelo.wv]
    return np.mean(vectores, axis=0) if vectores else np.zeros(modelo.vector_size)

def vectorize(tokens, modelo):
    """
    Alternativa a vector_promedio. Calcula el vector promedio para los tokens usando el modelo dado.
    """
    vecs = [modelo.wv[word] for word in tokens if word in modelo.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(modelo.vector_size)