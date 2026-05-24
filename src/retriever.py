import chromadb

from .config import (
    GROQ_API_KEY, MODO, MODELO_GROQ, MODELO_LLM,
    N_QUERY_VARIANTS, N_RESULTADOS, SIMILARITY_THRESHOLD,
    console, embedding_fn,
)


def _generar_variantes(pregunta: str) -> list[str]:
    prompt = (
        f"Genera {N_QUERY_VARIANTS} reformulaciones diferentes de la siguiente pregunta "
        f"para mejorar la búsqueda en documentos. "
        f"Responde SOLO con las preguntas reformuladas, una por línea, sin numeración ni explicaciones.\n\n"
        f"Pregunta original: {pregunta}"
    )
    try:
        if MODO == "local":
            import ollama
            resp = ollama.chat(model=MODELO_LLM, messages=[{"role": "user", "content": prompt}])
            texto = resp["message"]["content"]
        else:
            from groq import Groq
            cliente = Groq(api_key=GROQ_API_KEY)
            resp = cliente.chat.completions.create(
                model=MODELO_GROQ,
                messages=[{"role": "user", "content": prompt}],
            )
            texto = resp.choices[0].message.content

        return [l.strip() for l in texto.strip().splitlines() if l.strip()][:N_QUERY_VARIANTS]
    except Exception:
        return []


def buscar_contexto(coleccion: chromadb.Collection, pregunta: str) -> tuple[list[str], list[dict]]:
    console.print("[dim]Generando variantes de búsqueda...[/dim]")
    variantes = [pregunta] + _generar_variantes(pregunta)

    vistos: set[str] = set()
    docs_final: list[str] = []
    metas_final: list[dict] = []

    for variante in variantes:
        emb = embedding_fn.embed_query(variante)
        res = coleccion.query(
            query_embeddings=[emb],
            n_results=N_RESULTADOS,
            include=["documents", "metadatas", "distances"],
        )
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            if dist > SIMILARITY_THRESHOLD or doc in vistos:
                continue
            vistos.add(doc)
            docs_final.append(doc)
            metas_final.append(meta)

    return docs_final, metas_final
