import re

import chromadb

from .config import (
    GROQ_API_KEY, MODO, MODELO_GROQ, MODELO_LLM,
    N_QUERY_VARIANTS, N_RESULTADOS, SIMILARITY_THRESHOLD,
    console, embedding_fn,
)

MAX_HISTORIAL_REESCRITURA = 4

_TIPOS_INTERNOS = ["politica", "procedimiento", "documento_interno"]

# Palabras clave para detectar intención de la pregunta
_RE_NORMA = re.compile(
    r'\biso\b|iso[\s\-]?27\d{3}|27001|27002'
    r'|\bnorma\b|\bnormativa\b|\bestándar\b'
    r'|\bcontrol\s+\d|\bcontrol\s+[A-Z]\.\d'
    r'|\bcláusula\s+\d|\brequisito\s+de\s+la\s+norma\b',
    re.IGNORECASE,
)
_RE_INTERNA = re.compile(
    r'\bmi\s+empresa\b|\bmi\s+organizaci[oó]n\b'
    r'|\bnuestra[s]?\b|\bnuestro[s]?\b'
    r'|\bdocumentaci[oó]n\s+interna\b|\bpol[ií]tica\s+(de|interna)\b'
    r'|\bprocedimiento\s+(de|interno)\b|\bmarmotech\b',
    re.IGNORECASE,
)
_RE_COMPARAR = re.compile(
    r'\bcumpl[ei]\b|\bcumplimiento\b|\bgap\b|\bbrecha[s]?\b'
    r'|\bcompar[ae]\b|\bconforme\b|\bverifica\b|\banaliz[ae]\b'
    r'|\bcumple\s+con\b|\bsatisface\b|\bse\s+ajusta\b|\badecuado\b'
    r'|\bcumple[n]?\s+(mis|nuestros?|la|los|con)',
    re.IGNORECASE,
)


def detectar_intencion(pregunta: str) -> str:
    """Clasifica la pregunta en: 'norma', 'interna', 'comparacion' o 'general'."""
    kw_norma = bool(_RE_NORMA.search(pregunta))
    kw_interna = bool(_RE_INTERNA.search(pregunta))
    kw_comparar = bool(_RE_COMPARAR.search(pregunta))

    if kw_comparar or (kw_norma and kw_interna):
        return "comparacion"
    if kw_norma:
        return "norma"
    if kw_interna:
        return "interna"
    return "general"


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
        "Reescribe la última pregunta del usuario de forma autónoma, incluyendo los nombres "
        "propios o entidades necesarias del contexto. "
        "Si la pregunta ya es autónoma, devuélvela tal cual. "
        "Responde SOLO con la pregunta reescrita, sin explicaciones.\n\n"
        "EJEMPLO:\n"
        "HISTORIAL:\n"
        "USUARIO: ¿Cuáles son los niveles de clasificación de mi empresa?\n"
        "ASISTENTE: Los niveles son PACMAN, MARIO, POKEMON y HARRY POTTER...\n"
        "PREGUNTA ORIGINAL: Cuéntame más sobre el más restrictivo\n"
        "PREGUNTA REESCRITA: Cuéntame más sobre el nivel de clasificación HARRY POTTER\n\n"
        "AHORA HAZLO PARA:\n"
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
                max_tokens=200,
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


def _ejecutar_variantes(
    coleccion: chromadb.Collection,
    variantes: list[str],
    where: dict | None = None,
) -> tuple[list[str], list[dict]]:
    """Lanza las variantes contra la colección con el filtro indicado."""
    vistos: set[str] = set()
    docs_final: list[str] = []
    metas_final: list[dict] = []

    for variante in variantes:
        emb = embedding_fn.embed_query(variante)
        kwargs: dict = dict(
            query_embeddings=[emb],
            n_results=N_RESULTADOS,
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where
        res = coleccion.query(**kwargs)
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            if dist > SIMILARITY_THRESHOLD or doc in vistos:
                continue
            vistos.add(doc)
            docs_final.append(doc)
            metas_final.append(meta)

    return docs_final, metas_final


def buscar_contexto(
    coleccion: chromadb.Collection,
    pregunta: str,
    historial: list[dict] | None = None,
    intencion: str = "general",
) -> tuple[list[str], list[dict]]:
    pregunta_busqueda = _reescribir_con_contexto(pregunta, historial or [])
    if pregunta_busqueda != pregunta:
        console.print(f"[dim]Pregunta reformulada: {pregunta_busqueda}[/dim]")

    console.print("[dim]Generando variantes de búsqueda...[/dim]")
    variantes = [pregunta_busqueda] + _generar_variantes(pregunta_busqueda)

    if intencion == "norma":
        return _ejecutar_variantes(coleccion, variantes, {"tipo_doc": "norma_iso"})

    if intencion == "interna":
        return _ejecutar_variantes(
            coleccion, variantes, {"tipo_doc": {"$in": _TIPOS_INTERNOS}}
        )

    if intencion == "comparacion":
        # Búsqueda separada para garantizar resultados de ambas fuentes
        chunks_n, metas_n = _ejecutar_variantes(
            coleccion, variantes, {"tipo_doc": "norma_iso"}
        )
        chunks_i, metas_i = _ejecutar_variantes(
            coleccion, variantes, {"tipo_doc": {"$in": _TIPOS_INTERNOS}}
        )
        return chunks_n + chunks_i, metas_n + metas_i

    # general: sin filtro
    return _ejecutar_variantes(coleccion, variantes, None)
