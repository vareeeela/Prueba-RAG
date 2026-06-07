import re

import chromadb

from .config import (
    GROQ_API_KEY, MODO, MODELO_GROQ, MODELO_LLM,
    N_QUERY_VARIANTS, N_RESULTADOS, SIMILARITY_THRESHOLD,
    console, embedding_fn,
)

MAX_HISTORIAL_REESCRITURA = 4
N_MINIMO_POR_LADO_COMPARACION = 3

_TIPOS_INTERNOS = ["politica", "procedimiento", "documento_interno"]

# Palabras clave para detectar intención de la pregunta
_RE_NORMA = re.compile(
    r'\biso\b'
    r'|iso[\s\-]?27\d{3}'
    r'|\b27001\b|\b27002\b'
    r'|\bnorma\b'
    r'|\bnormativa\b(?!\s+interna)'
    r'|\bestándar(?:es)?\b(?!\s*[\?\.,;:]|\s+(de\s+)?(usuario|usuarios|cuenta|cuentas))'
    r'|\bestandar(?:es)?\b(?!\s*[\?\.,;:]|\s+(de\s+)?(usuario|usuarios|cuenta|cuentas))'
    r'|\bcontrol\s+\d|\bcontrol\s+[A-Z]\.\d'
    r'|\bcláusula\s+\d|\bclausula\s+\d'
    r'|\brequisito\s+de\s+la\s+norma\b',
    re.IGNORECASE,
)
_RE_INTERNA = re.compile(
    r'\bmi\s+empresa\b'
    r'|\bmi\s+organizaci[oó]n\b'
    r'|\bmi\s+(política|politica|normativa|procedimiento)\b'
    r'|\bnuestra[s]?\b|\bnuestro[s]?\b'
    r'|\bdocumentaci[oó]n\s+interna\b'
    r'|\bpolítica\s+interna\b|\bpolitica\s+interna\b'
    r'|\bnormativa\s+interna\b'
    r'|\bprocedimiento\s+interno\b'
    r'|\bmarmotech\b',
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


def _generar_variantes(pregunta: str, orientacion: str = "neutra") -> list[str]:
    """
    Genera N_QUERY_VARIANTS reformulaciones de la pregunta para mejorar el retrieval.

    orientacion controla el sesgo léxico de las variantes:
      - "neutra":   reformulaciones generales (modos norma, interna y general).
      - "norma":    variantes orientadas a la normativa ISO (control, cláusula, requisito).
      - "interna":  variantes orientadas a documentación interna (datos operativos concretos).
    """
    if orientacion == "norma":
        prompt = (
            f"Genera {N_QUERY_VARIANTS} reformulaciones de la siguiente pregunta orientadas "
            f"a recuperar información de NORMAS ISO/IEC (controles, cláusulas, requisitos). "
            f"Usa vocabulario normativo: 'control', 'cláusula', 'requisito', 'exige', 'recomienda'. "
            f"Responde SOLO con las preguntas reformuladas, una por línea, sin numeración ni explicaciones.\n\n"
            f"Pregunta original: {pregunta}"
        )
    elif orientacion == "interna":
        prompt = (
            f"Genera {N_QUERY_VARIANTS} reformulaciones de la siguiente pregunta orientadas "
            f"a recuperar el dato operativo concreto en POLÍTICAS o PROCEDIMIENTOS INTERNOS de una empresa. "
            f"Usa términos concretos del dominio: parámetros numéricos, plazos, longitudes, nombres de sistemas, "
            f"responsables, frecuencias, valores específicos. "
            f"Evita vocabulario abstracto como 'recomendación', 'estándar', 'norma' o 'control'. "
            f"Responde SOLO con las preguntas reformuladas, una por línea, sin numeración ni explicaciones.\n\n"
            f"Pregunta original: {pregunta}"
        )
    else:
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
    ignorar_umbral: bool = False,
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
            if (not ignorar_umbral and dist > SIMILARITY_THRESHOLD) or doc in vistos:
                continue
            vistos.add(doc)
            docs_final.append(doc)
            meta_entry = dict(meta)
            meta_entry["distancia"] = dist
            metas_final.append(meta_entry)

    return docs_final, metas_final


def _log_chunks(docs: list[str], metas: list[dict]) -> None:
    console.print(f"[dim][retrieval] {len(docs)} chunks recuperados:[/dim]")
    for i, meta in enumerate(metas, 1):
        dist = meta.get("distancia")
        dist_str = f"{dist:.3f}" if isinstance(dist, float) else "?"
        console.print(
            f"[dim]  [{i}] {meta.get('fuente', '?')} "
            f"cláusula={meta.get('clausula', '-')} "
            f"dist={dist_str}[/dim]"
        )


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

    if intencion == "norma":
        variantes = [pregunta_busqueda] + _generar_variantes(pregunta_busqueda, orientacion="norma")
        docs, metas = _ejecutar_variantes(coleccion, variantes, {"tipo_doc": "norma_iso"})
        _log_chunks(docs, metas)
        return docs, metas

    if intencion == "interna":
        variantes = [pregunta_busqueda] + _generar_variantes(pregunta_busqueda, orientacion="interna")
        docs, metas = _ejecutar_variantes(
            coleccion, variantes, {"tipo_doc": {"$in": _TIPOS_INTERNOS}}
        )
        _log_chunks(docs, metas)
        return docs, metas

    if intencion == "comparacion":
        # Dos conjuntos independientes de variantes, sesgados a cada lado
        variantes_norma = [pregunta_busqueda] + _generar_variantes(
            pregunta_busqueda, orientacion="norma"
        )
        variantes_interna = [pregunta_busqueda] + _generar_variantes(
            pregunta_busqueda, orientacion="interna"
        )

        chunks_n, metas_n = _ejecutar_variantes(
            coleccion, variantes_norma, {"tipo_doc": "norma_iso"}
        )
        chunks_i, metas_i = _ejecutar_variantes(
            coleccion, variantes_interna, {"tipo_doc": {"$in": _TIPOS_INTERNOS}}
        )

        # Garantizar mínimo por cada lado aunque queden bajo el umbral
        if len(chunks_n) < N_MINIMO_POR_LADO_COMPARACION:
            fb_n, fb_metas_n = _ejecutar_variantes(
                coleccion, variantes_norma, {"tipo_doc": "norma_iso"}, ignorar_umbral=True
            )
            vistos_n = set(chunks_n)
            faltan = N_MINIMO_POR_LADO_COMPARACION - len(chunks_n)
            for c, m in zip(fb_n, fb_metas_n):
                if c not in vistos_n and faltan > 0:
                    chunks_n.append(c)
                    metas_n.append(m)
                    vistos_n.add(c)
                    faltan -= 1

        if len(chunks_i) < N_MINIMO_POR_LADO_COMPARACION:
            fb_i, fb_metas_i = _ejecutar_variantes(
                coleccion, variantes_interna,
                {"tipo_doc": {"$in": _TIPOS_INTERNOS}}, ignorar_umbral=True,
            )
            vistos_i = set(chunks_i)
            faltan = N_MINIMO_POR_LADO_COMPARACION - len(chunks_i)
            for c, m in zip(fb_i, fb_metas_i):
                if c not in vistos_i and faltan > 0:
                    chunks_i.append(c)
                    metas_i.append(m)
                    vistos_i.add(c)
                    faltan -= 1

        docs = chunks_n + chunks_i
        metas = metas_n + metas_i
        _log_chunks(docs, metas)
        return docs, metas

    # general: sin filtro, variantes neutras
    variantes = [pregunta_busqueda] + _generar_variantes(pregunta_busqueda, orientacion="neutra")
    docs, metas = _ejecutar_variantes(coleccion, variantes, None)
    _log_chunks(docs, metas)
    return docs, metas
