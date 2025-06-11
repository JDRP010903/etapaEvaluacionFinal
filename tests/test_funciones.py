import unittest
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

from funciones import limpiar_tokens, limpiar_texto, vector_promedio, vectorize

# Descargar recursos necesarios de NLTK
nltk.download('punkt')
nltk.download('stopwords')

class TestFuncionesEmbeddings(unittest.TestCase):

    def setUp(self):
        # Crear un modelo Word2Vec de prueba
        self.modelo = Word2Vec(sentences=[["anorexia", "salud", "mental"]], vector_size=100, min_count=1)

    def test_limpiar_tokens_basico(self):
        texto = "Hola @user, visita https://ejemplo.com #salud123!"
        resultado = limpiar_tokens(texto)
        self.assertIsInstance(resultado, list)
        self.assertTrue(all(isinstance(token, str) for token in resultado))
        self.assertNotIn("user", resultado)
        self.assertGreater(len(resultado), 0)

    def test_limpiar_tokens_vacio(self):
        resultado = limpiar_tokens("")
        self.assertEqual(resultado, [])

    def test_limpiar_texto(self):
        texto = "#Anorexia #mental afecta a jóvenes"
        resultado = limpiar_texto(texto)
        self.assertIsInstance(resultado, str)
        self.assertGreater(len(resultado), 0)

    def test_vector_promedio_normal(self):
        tokens = ["anorexia", "salud"]
        vec = vector_promedio(tokens, self.modelo)
        self.assertEqual(vec.shape, (100,))
        self.assertFalse(np.isnan(vec).any())

    def test_vector_promedio_vacio(self):
        vec = vector_promedio([], self.modelo)
        self.assertEqual(vec.shape, (100,))
        self.assertTrue(np.all(vec == 0))

    def test_vectorize_normal(self):
        tokens = ["anorexia", "mental"]
        vec = vectorize(tokens, self.modelo)
        self.assertEqual(vec.shape, (100,))
        self.assertFalse(np.isnan(vec).any())

    def test_vectorize_vacio(self):
        vec = vectorize([], self.modelo)
        self.assertEqual(vec.shape, (100,))
        self.assertTrue(np.all(vec == 0))

if __name__ == '__main__':
    unittest.main()