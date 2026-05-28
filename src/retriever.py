import chromadb

from .config import (
    GROQ_API_KEY, MODO, MODELO_GROQ, MODELO_LLM,
    N_QUERY_VARIANTS, N_RESULTADOS, SIMILARITY_THRESHOLD,
    console, embedding_fn,
)

MAX_HISTORIAL_REESCRITURA = 4  # mensajes recientes que se dan como contexto al reescritor


def _reescribir_con_contexto(pregunta: str, historial: list[dict]) -> str:
    if not historial:
        return pregunta

    ultimos = [m for m in historial[-MAX_HISTORIAL_REESCRITURA:] if m["rol"] in ("user", "assistant")]
    if not ultimos:
        return pregunta

    historial_str = "\n".join(
        f"{'USUARIO' if m['rol'] == 'user' else 'ASISTENTE'}: {m['contenido'][:300]}"
        for m in ultimos
    )
    prompt = (
        "Dado el historial de conversación siguiente, reescribe la última pregunta del usuario "
        "de forma que sea completamente autónoma y comprensible sin el historial "
        "(incluye los nombres propios o entidades necesarias del contexto). "
        "Si la pregunta ya es autónoma, devuélvela tal cual. "
        "Responde SOLO con la pregunta reescrita, sin explicaciones.\n\n"
        f"HISTORIAL:\n{historial_str}\n\n"
        f"PREGUNTA ORIGINAL: {pregunta}\n"
        "PREGUNTA REESCRITA:"
    )
    try:
        if MODO == "local":
            import ollama
            resp = ollama.chat(model=MODELO_LLM, messages=[{"role": "user", "content": prompt}])
            reescrita = resp["message"]["content"].strip()
        else:
            from groq import Groq
            resp = Groq(api_key=GROQ_API_KEY).chat.completions.create(
                model=MODELO_GROQ,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=120,
            )
            reescrita = resp.choices[0].message.content.strip()
        return reescrita or pregunta
    except Exception:
        return pregunta


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


def buscar_contexto(
    coleccion: chromadb.Collection,
    pregunta: str,
    historial: list[dict] | None = None,
) -> tuple[list[str], list[dict]]:
    pregunta_busqueda = _reescribir_con_contexto(pregunta, historial or [])
    if pregunta_busqueda != pregunta:
        console.print(f"[dim]Pregunta reformulada: {pregunta_busqueda}[/dim]")

    console.print("[dim]Generando variantes de búsqueda...[/dim]")
    variantes = [pregunta_busqueda] + _generar_variantes(pregunta_busqueda)

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
