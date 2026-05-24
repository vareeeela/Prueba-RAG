from sentence_transformers import SentenceTransformer

# Cargamos un modelo de embeddings gratuito (se descarga solo la primera vez)
modelo = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Texto de prueba
textos = [
    "El control A.8.8 trata sobre la gestión de vulnerabilidades técnicas.",
    "La política de contraseñas exige un mínimo de 12 caracteres.",
    "El responsable del SGSI debe revisar los controles anualmente."
]

# Generamos los embeddings
embeddings = modelo.encode(textos)

print(f"Número de fragmentos: {len(embeddings)}")
print(f"Dimensiones de cada embedding: {len(embeddings[0])}")
print("✅ Embeddings generados correctamente")