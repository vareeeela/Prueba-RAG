import re
from collections import Counter

import chromadb

from .config import (
    GROQ_API_KEY, MODO, MODELO_GROQ, MODELO_LLM,
    N_QUERY_VARIANTS, N_RESULTADOS, SIMILARITY_THRESHOLD,
    console, embedding_fn,
)

MAX_HISTORIAL_REESCRITURA = 4
N_MINIMO_POR_LADO_COMPARACION = 3
MAX_CONTEXT_CHUNKS = 20          # tope global de chunks enviados al modelo
MAX_CONTEXT_CHUNKS_POR_LADO = 10 # para modo comparacion (cada lado norma/interna)
# Umbral blando para el fallback interna: evita ruido extremo al forzar el mínimo
_UMBRAL_FALLBACK = SIMILARITY_THRESHOLD + 0.20

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
    r'|\bmarmotech\b'
    # Preguntas de aplicación práctica de políticas internas
    r'|\b(con\s+qu[eé]|qu[eé])\s+etiqueta[s]?\b'
    r'|\bqu[eé]\s+(nivel|categor[íi]a|tipo)\s+(de\s+)?(clasificaci[oó]n|confidencialidad)\b'
    r'|\bc[oó]mo\s+(se\s+)?(clasifico|etiqueto|clasifica|etiqueta|clasificar[íi]a?|etiquetar[íi]a?)\b'
    r'|\bdebo\s+(etiquetar|clasificar|tratar|gestionar)\b'
    r'|\btengo\s+que\s+(etiquetar|clasificar)\b'
    r'|\bqu[eé]\s+clasificaci[oó]n\s+(le\s+)?(corresponde|aplica|debo|tengo)\b'
    r'|\bcontrase[ñn]a[s]?\b|\blongitud\s+(m[íi]nima|m[aá]xima)\b'
    r'|\bplazo[s]?\s+de\b|\bcad[eu]ca\b|\bvigencia\b'
    r'|\bresponsable[s]?\s+de\b|\bquién\s+(es\s+)?(responsible|encargado)\b',
    re.IGNORECASE,
)
_RE_CLAUSULA_NORMA = re.compile(
    r'(?:punto|p[uú]nto|cl[aá]usula|secci[oó]n|apartado|art[íi]culo|control)\s+(\d+(?:\.\d+)*)',
    re.IGNORECASE,
)
_RE_CLASIFICACION = re.compile(
    r"etiqueta[s]?\b"
    r"|qu[eé]\s+(?:nivel|clasificaci[oó]n)\b"
    r"|c[oó]mo\s+(?:se\s+)?(?:clasifico|etiqueto|clasificar|etiquetar)\b"
    r"|debo\s+(?:etiquetar|clasificar)\b"
    r"|nivel\s+de\s+clasificaci[oó]n\b"
    r"|qu[eé]\s+clasificaci[oó]n\b",
    re.IGNORECASE,
)
_RE_COMPARAR = re.compile(
    r'\bcumpl[ei]\b|\bcumplimiento\b|\bgap\b|\bbrecha[s]?\b'
    r'|\bcompar[ae]\b|\bconforme\b|\bverifica\b|\banaliz[ae]\b'
    r'|\bcumple\s+con\b|\bsatisface\b|\bse\s+ajusta\b|\badecuado\b'
    r'|\bcumple[n]?\s+(mis|nuestros?|la|los|con)'
    r'|\bcubr[aeo]\b|\bcubierto\b',
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
                max_tokens=400,
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
                max_tokens=400,
            )
            texto = resp.choices[0].message.content

        return [l.strip() for l in texto.strip().splitlines() if l.strip()][:N_QUERY_VARIANTS]
    except Exception:
        return []


def _generar_hyde(pregunta: str) -> str | None:
    """Genera un fragmento de documento hipotético (HyDE) para mejorar el retrieval.

    El texto generado se embebe con prefijo 'passage:' para que su vector sea
    comparable directamente con los documentos indexados, cerrando la brecha
    semántica entre la pregunta (espacio de queries) y la respuesta (espacio de pasajes).
    """
    prompt = (
        "Escribe un fragmento breve (3-5 líneas) de un documento de política de seguridad "
        "de la información que contendría la respuesta a la siguiente pregunta. "
        "Usa terminología técnica concreta. Solo el texto del fragmento, sin encabezado.\n\n"
        f"Pregunta: {pregunta}"
    )
    try:
        if MODO == "local":
            import ollama
            resp = ollama.chat(
                model=MODELO_LLM,
                messages=[{"role": "user", "content": prompt}],
                options={"num_predict": 120},
            )
            return resp["message"]["content"].strip()
        else:
            from groq import Groq
            resp = Groq(api_key=GROQ_API_KEY).chat.completions.create(
                model=MODELO_GROQ,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
    except Exception:
        return None


def _ejecutar_hyde(
    coleccion: chromadb.Collection,
    hyde_text: str,
    where: dict | None = None,
    vistos: set[str] | None = None,
) -> tuple[list[str], list[dict]]:
    """Busca con el embedding passage: del texto hipotético y actualiza vistos."""
    if vistos is None:
        vistos = set()
    hyde_emb = embedding_fn([hyde_text])[0]  # prefijo passage:, como los docs indexados
    kwargs: dict = dict(
        query_embeddings=[hyde_emb],
        n_results=N_RESULTADOS,
        include=["documents", "metadatas", "distances"],
    )
    if where:
        kwargs["where"] = where
    res = coleccion.query(**kwargs)
    docs_final: list[str] = []
    metas_final: list[dict] = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        if dist > SIMILARITY_THRESHOLD or doc in vistos:
            continue
        vistos.add(doc)
        docs_final.append(doc)
        meta_entry = dict(meta)
        meta_entry["distancia"] = dist
        metas_final.append(meta_entry)
    return docs_final, metas_final


def _ejecutar_variantes(
    coleccion: chromadb.Collection,
    variantes: list[str],
    where: dict | None = None,
    ignorar_umbral: bool = False,
    vistos: set[str] | None = None,
) -> tuple[list[str], list[dict]]:
    """Lanza las variantes contra la colección con el filtro indicado."""
    if vistos is None:
        vistos = set()
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


def _forzar_clausulas_norma(
    coleccion: chromadb.Collection,
    clausulas: list[str],
    vistos: set[str],
) -> tuple[list[str], list[dict]]:
    """Recupera cláusulas ISO por metadata cuando se mencionan explícitamente en la pregunta."""
    try:
        res = coleccion.get(
            where={"$and": [
                {"tipo_doc": {"$eq": "norma_iso"}},
                {"clausula": {"$in": clausulas}},
            ]},
            include=["documents", "metadatas"],
        )
        docs, metas = [], []
        for doc, meta in zip(res.get("documents", []), res.get("metadatas", [])):
            if doc in vistos:
                continue
            vistos.add(doc)
            docs.append(doc)
            metas.append({**meta, "distancia": 0.0})
        return docs, metas
    except Exception:
        return [], []


def _forzar_secciones_documento(
    coleccion: chromadb.Collection,
    fuente: str,
    clausulas: list[str],
    vistos: set[str],
) -> tuple[list[str], list[dict]]:
    """Recupera secciones específicas de un documento por metadata exacta.

    Se usa como fallback cuando la búsqueda semántica no recupera secciones
    que sabemos que existen por su clausula pero cuyo vocabulario no coincide
    con la consulta del usuario (brecha semántica de terminología específica).
    """
    try:
        res = coleccion.get(
            where={"$and": [
                {"fuente": {"$eq": fuente}},
                {"clausula": {"$in": clausulas}},
            ]},
            include=["documents", "metadatas"],
        )
        docs, metas = [], []
        for doc, meta in zip(res.get("documents", []), res.get("metadatas", [])):
            if doc in vistos:
                continue
            vistos.add(doc)
            docs.append(doc)
            metas.append({**meta, "distancia": 0.0})
        return docs, metas
    except Exception:
        return [], []


def _recortar_contexto(
    docs: list[str], metas: list[dict], max_chunks: int
) -> tuple[list[str], list[dict]]:
    """Ordena por distancia y recorta al máximo de chunks permitido."""
    if len(docs) <= max_chunks:
        return docs, metas
    paired = sorted(zip(docs, metas), key=lambda x: x[1].get("distancia", 1.0))
    return [d for d, _ in paired[:max_chunks]], [m for _, m in paired[:max_chunks]]


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

    # Fragmento hipotético (HyDE): se embebe con prefijo passage: para buscar
    # en el espacio semántico de los documentos, no en el de las preguntas.
    # Los resultados se anteponen a los de las variantes de query normales.
    hyde_text = _generar_hyde(pregunta_busqueda)
    if hyde_text:
        console.print("[dim][HyDE] Fragmento hipotético generado[/dim]")

    if intencion == "norma":
        filtro = {"tipo_doc": "norma_iso"}
        vistos: set[str] = set()
        hyde_docs, hyde_metas = _ejecutar_hyde(coleccion, hyde_text, filtro, vistos) if hyde_text else ([], [])
        variantes = [pregunta_busqueda] + _generar_variantes(pregunta_busqueda, orientacion="norma")
        docs, metas = _ejecutar_variantes(coleccion, variantes, filtro, vistos=vistos)
        docs, metas = hyde_docs + docs, hyde_metas + metas

        # Fallback: si la pregunta menciona un número de cláusula, forzar su recuperación
        clausulas_mencionadas = _RE_CLAUSULA_NORMA.findall(pregunta_busqueda)
        if clausulas_mencionadas:
            extra_docs, extra_metas = _forzar_clausulas_norma(
                coleccion, clausulas_mencionadas, vistos
            )
            if extra_docs:
                docs = extra_docs + docs
                metas = extra_metas + metas

        docs, metas = _recortar_contexto(docs, metas, MAX_CONTEXT_CHUNKS)
        _log_chunks(docs, metas)
        return docs, metas

    if intencion == "interna":
        filtro = {"tipo_doc": {"$in": _TIPOS_INTERNOS}}
        vistos: set[str] = set()
        hyde_docs, hyde_metas = _ejecutar_hyde(coleccion, hyde_text, filtro, vistos) if hyde_text else ([], [])
        variantes = [pregunta_busqueda] + _generar_variantes(pregunta_busqueda, orientacion="interna")
        docs, metas = _ejecutar_variantes(coleccion, variantes, filtro, vistos=vistos)
        docs, metas = hyde_docs + docs, hyde_metas + metas

        # Fallback de esquemas: si la pregunta es sobre clasificación y ya recuperamos
        # alguna sección 5.x pero faltan subsecciones, forzar su recuperación por metadata.
        # Resuelve la brecha semántica entre terminología del usuario y vocabulario del documento.
        if _RE_CLASIFICACION.search(pregunta_busqueda):
            clausulas_ya = {str(m.get("clausula", "")) for m in metas}
            fuentes_5x = [
                m.get("fuente") for m in metas
                if str(m.get("clausula", "")).startswith("5") and m.get("fuente")
            ]
            if fuentes_5x and not {"5.1", "5.2", "5.3"}.issubset(clausulas_ya):
                fuente_cls = Counter(fuentes_5x).most_common(1)[0][0]
                extra_docs, extra_metas = _forzar_secciones_documento(
                    coleccion, fuente_cls,
                    ["5", "5.1", "5.2", "5.3", "5.4", "5.5"], vistos,
                )
                if extra_docs:
                    docs = extra_docs + docs
                    metas = extra_metas + metas

        docs, metas = _recortar_contexto(docs, metas, MAX_CONTEXT_CHUNKS)
        _log_chunks(docs, metas)
        return docs, metas

    if intencion == "comparacion":
        variantes_norma = [pregunta_busqueda] + _generar_variantes(
            pregunta_busqueda, orientacion="norma"
        )
        variantes_interna = [pregunta_busqueda] + _generar_variantes(
            pregunta_busqueda, orientacion="interna"
        )

        # Lado norma: HyDE + variantes
        filtro_n = {"tipo_doc": "norma_iso"}
        vistos_n: set[str] = set()
        hyde_n, hyde_metas_n = _ejecutar_hyde(coleccion, hyde_text, filtro_n, vistos_n) if hyde_text else ([], [])
        chunks_n, metas_n = _ejecutar_variantes(coleccion, variantes_norma, filtro_n, vistos=vistos_n)
        chunks_n, metas_n = hyde_n + chunks_n, hyde_metas_n + metas_n

        # Lado interna: HyDE + variantes (vistos independiente para no contaminar cruce)
        filtro_i = {"tipo_doc": {"$in": _TIPOS_INTERNOS}}
        vistos_i: set[str] = set()
        hyde_i, hyde_metas_i = _ejecutar_hyde(coleccion, hyde_text, filtro_i, vistos_i) if hyde_text else ([], [])
        chunks_i, metas_i = _ejecutar_variantes(coleccion, variantes_interna, filtro_i, vistos=vistos_i)
        chunks_i, metas_i = hyde_i + chunks_i, hyde_metas_i + metas_i

        # Búsqueda contextual cruzada: usa el texto de los chunks norma recuperados
        # como embedding passage para buscar el equivalente en docs internos.
        # Cierra la brecha semántica cuando la query original es anafórica o corta
        # y la sección interna relevante usa vocabulario distinto al de la query.
        if chunks_n:
            norma_ctx = " ".join(chunks_n[:2])[:800]
            ctx_docs, ctx_metas = _ejecutar_hyde(coleccion, norma_ctx, filtro_i, vistos_i)
            if ctx_docs:
                chunks_i = ctx_docs + chunks_i
                metas_i = ctx_metas + metas_i

        # Garantizar mínimo por cada lado aunque queden bajo el umbral
        if len(chunks_n) < N_MINIMO_POR_LADO_COMPARACION:
            fb_n, fb_metas_n = _ejecutar_variantes(
                coleccion, variantes_norma, filtro_n, ignorar_umbral=True, vistos=set(chunks_n)
            )
            faltan = N_MINIMO_POR_LADO_COMPARACION - len(chunks_n)
            vistos_extra = set(chunks_n)
            for c, m in zip(fb_n, fb_metas_n):
                if c not in vistos_extra and faltan > 0:
                    chunks_n.append(c)
                    metas_n.append(m)
                    vistos_extra.add(c)
                    faltan -= 1

        if len(chunks_i) < N_MINIMO_POR_LADO_COMPARACION:
            fb_i, fb_metas_i = _ejecutar_variantes(
                coleccion, variantes_interna, filtro_i, ignorar_umbral=True, vistos=set(chunks_i)
            )
            faltan = N_MINIMO_POR_LADO_COMPARACION - len(chunks_i)
            vistos_extra = set(chunks_i)
            for c, m in zip(fb_i, fb_metas_i):
                if (
                    c not in vistos_extra
                    and faltan > 0
                    and m.get("distancia", 1.0) <= _UMBRAL_FALLBACK
                ):
                    chunks_i.append(c)
                    metas_i.append(m)
                    vistos_extra.add(c)
                    faltan -= 1

        chunks_n, metas_n = _recortar_contexto(chunks_n, metas_n, MAX_CONTEXT_CHUNKS_POR_LADO)
        chunks_i, metas_i = _recortar_contexto(chunks_i, metas_i, MAX_CONTEXT_CHUNKS_POR_LADO)
        docs = chunks_n + chunks_i
        metas = metas_n + metas_i
        _log_chunks(docs, metas)
        return docs, metas

    # general: HyDE + variantes neutras + interna, sin filtro de tipo
    vistos: set[str] = set()
    hyde_docs, hyde_metas = _ejecutar_hyde(coleccion, hyde_text, None, vistos) if hyde_text else ([], [])
    variantes = (
        [pregunta_busqueda]
        + _generar_variantes(pregunta_busqueda, orientacion="neutra")
        + _generar_variantes(pregunta_busqueda, orientacion="interna")
    )
    docs, metas = _ejecutar_variantes(coleccion, variantes, None, vistos=vistos)
    docs, metas = hyde_docs + docs, hyde_metas + metas
    docs, metas = _recortar_contexto(docs, metas, MAX_CONTEXT_CHUNKS)
    _log_chunks(docs, metas)
    return docs, metas
