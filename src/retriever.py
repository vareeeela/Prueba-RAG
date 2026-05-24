import chromadb

from .config import N_RESULTADOS


def buscar_contexto(coleccion: chromadb.Collection, pregunta: str) -> tuple[list, list]:
    res = coleccion.query(
        query_texts=[pregunta],
        n_results=N_RESULTADOS,
        include=["documents", "metadatas", "distances"],
    )
    docs = res["documents"][0]
    fuentes = [m["fuente"] for m in res["metadatas"][0]]
    return docs, fuentes
