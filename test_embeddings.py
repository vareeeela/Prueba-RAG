from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
modelo = SentenceTransformer(MODEL_NAME)

fragmentos = [
    "El control A.8.8 trata sobre la gestión de vulnerabilidades técnicas.",
    "La política de contraseñas exige un mínimo de 12 caracteres.",
    "El responsable del SGSI debe revisar los controles anualmente.",
]
consulta = "¿Cómo se gestionan las vulnerabilidades de seguridad?"

todos = fragmentos + [consulta]
embeddings = modelo.encode(todos, normalize_embeddings=True)
emb_fragmentos = embeddings[:-1]
emb_consulta = embeddings[-1:]

print(f"Modelo : {MODEL_NAME}")
print(f"Dimensiones: {embeddings.shape[1]}\n")
print(f'Consulta: "{consulta}"\n')
print("Similitud coseno con cada fragmento:")
for texto, sim in zip(fragmentos, cosine_similarity(emb_consulta, emb_fragmentos)[0]):
    print(f"  [{sim:.3f}]  {texto}")
print("\nEmbeddings generados correctamente.")
