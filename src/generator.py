import re
from typing import Iterator

import chromadb

from .config import (
    GROQ_API_KEY, MAX_TOKENS, MAX_TURNOS_HISTORIAL,
    MODO, MODELO_GROQ, MODELO_LLM, TEMPERATURE, console,
)
from .retriever import buscar_contexto, detectar_intencion

_PATRON_INYECCION = re.compile(
    r"olvida\s+(tus\s+)?(instrucciones|reglas|contexto|rol|restricciones|sistema)"
    r"|ignora\s+(tus\s+)?(instrucciones|reglas|restricciones|sistema)"
    r"|descarta\s+(tus\s+)?(instrucciones|reglas)"
    r"|(ahora\s+)?(eres|serás|actúa\s+como|compórtate\s+como|pretende\s+ser)\s+(?!lucIA)"
    r"|(forget|ignore)\s+(your\s+)?(instructions|rules|system\s+prompt|context)"
    r"|you\s+are\s+now\s+(?!lucIA)"
    r"|act\s+as\s+(?!an?\s+assistant)"
    r"|nueva[s]?\s+instrucciones?|new\s+instructions?"
    r"|modo\s+(sin\s+restricciones|libre|developer|dev|sin\s+límites)"
    r"|jailbreak|\bDAN\b"
    r"|olvida\s+que\s+eres\s+lucIA"
    r"|forget\s+(that\s+)?you\s+are\s+lucIA"
    r"|\[SYSTEM\]|\[INST\]|<\|system\|>"
    r"|(muéstrame?|repite|dame?|dime|revela|imprime|muestra)\s+(tod[ao]s?\s+)?(tus?\s+|las?\s+)?(el\s+)?(contexto|prompt|instrucciones?)"
    r"|(show|print|repeat|reveal|output|display)\s+(your\s+)?(context|system\s+prompt|instructions?|full\s+context)"
    r"|responde\s+con\s+todo",
    re.IGNORECASE,
)


def es_inyeccion_prompt(texto: str) -> bool:
    return bool(_PATRON_INYECCION.search(texto))


_ETIQUETAS_TIPO = {
    "norma_iso": "Norma ISO",
    "politica": "Política",
    "procedimiento": "Procedimiento",
    "documento_interno": "Doc. interno",
}


def _codigo_doc(fuente: str) -> str:
    """Extrae el código corto del nombre de fichero (p. ej. 'NOR-SEG-009' o 'ISO/IEC 27001')."""
    m = re.search(r'ISOIEC\s*(\d{5})\d{4}', fuente)
    if m:
        return f"ISO/IEC {m.group(1)}"
    m = re.match(r'([A-Z]{2,4}-[A-Z]{2,4}-\d{3})', fuente)
    if m:
        return m.group(1)
    return re.sub(r'\.[^.]+$', '', fuente).replace('_', ' ')


def _nombre_corto(fuente: str) -> str:
    """Convierte el nombre de fichero en una etiqueta legible."""
    m = re.search(r'ISOIEC\s*(\d{5})(\d{4})', fuente)
    if m:
        return f"ISO/IEC {m.group(1)}:{m.group(2)}"
    return re.sub(r'\.[^.]+$', '', fuente).replace('_', ' ')


def resumen_fuentes(metas: list[dict], neutro: bool = False) -> str:
    por_doc: dict[str, list] = {}
    for meta in metas:
        fuente = _nombre_corto(meta.get("fuente", ""))
        tipo = _ETIQUETAS_TIPO.get(meta.get("tipo_doc", ""), "")
        if neutro:
            clave = f"{fuente} ({tipo.lower()})" if tipo else fuente
        else:
            clave = f"{fuente} [{tipo}]" if tipo else fuente
        por_doc.setdefault(clave, [])
        ref = meta.get("clausula") or meta.get("seccion", "")
        if ref:
            ref_fmt = f"§{ref}"
            if ref_fmt not in por_doc[clave]:
                por_doc[clave].append(ref_fmt)

    partes = []
    for clave, refs in por_doc.items():
        partes.append(f"{clave} ({', '.join(refs)})" if refs else clave)

    sep = "; " if neutro else " | "
    cuerpo = sep.join(partes)
    return f"Fuentes consultadas: {cuerpo}" if neutro else cuerpo


def reemplazar_citas(respuesta: str, metas: list[dict]) -> str:
    """Sustituye [N] por [DOC-CODE §CLAUSE] para que las citas sean legibles."""
    def _sub(m: re.Match) -> str:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(metas):
            meta = metas[idx]
            codigo = _codigo_doc(meta.get("fuente", ""))
            clausula = meta.get("clausula", "")
            return f"[{codigo} §{clausula}]" if clausula else f"[{codigo}]"
        return m.group(0)

    return re.sub(r'\[(\d+)\]', _sub, respuesta)


def extraer_metas_citadas(respuesta: str, metas: list[dict], intencion: str) -> list[dict]:
    """Devuelve solo los metadatos de los chunks realmente citados en la respuesta.

    Estrategia en dos pasos:
    1. Citas numéricas [N] / [N1] / [I1] que el modelo añade inline.
    2. Si no hay citas numéricas, busca referencias a cláusulas nombradas
       en el texto ('Cláusula 5.3', 'punto 4.3', etc.) y filtra por ellas.
    """
    if intencion == "comparacion":
        n_total = sum(1 for m in metas if m.get("tipo_doc") == "norma_iso")
        n_refs = {int(x) - 1 for x in re.findall(r'\[N(\d+)\]', respuesta)}
        i_refs = {n_total + int(x) - 1 for x in re.findall(r'\[I(\d+)\]', respuesta)}
        indices = n_refs | i_refs
    else:
        indices = {int(x) - 1 for x in re.findall(r'\[(\d+)\]', respuesta) if x.isdigit()}

    if indices:
        return [m for i, m in enumerate(metas) if i in indices]

    # Fallback: referencias a cláusulas mencionadas en el texto
    clausulas_citadas = set(re.findall(
        r'\b(?:cl[aá]usula|punto|secci[oó]n|apartado)\s+(\d+(?:\.\d+)*)\b',
        respuesta, re.IGNORECASE,
    ))
    if clausulas_citadas:
        filtradas = [m for m in metas if m.get("clausula") in clausulas_citadas]
        if filtradas:
            return filtradas

    return metas


_STOPS = ["Pregunta:", "Usuario:", "\n¿", "\nAssistant:"]

_PATRON_IDENTIDAD = re.compile(
    r"^\s*[¿]?\s*(qu[eé]\s+eres|qui[eé]n\s+eres|qu[eé]\s+(?:puedes|haces|sabes|ofreces)(?:\s+hacer)?|"
    r"c[oó]mo\s+te\s+llamas|cu[aá]l\s+es\s+tu\s+(?:nombre|funci[oó]n|prop[oó]sito)|"
    r"para\s+qu[eé]\s+sirves?|qu[eé]\s+tipo\s+de\s+asistente|"
    r"d[ií]me\s+qu[eé]\s+eres|pres[eé]ntate)[.!?,¿\s]*$",
    re.IGNORECASE,
)

# Saludos: solo se exige que el mensaje EMPIECE con el saludo y sea corto (≤40 chars).
# Así "Buenas tardes" y "Hola, ¿qué tal?" no caen al RAG.
_PATRON_SALUDO = re.compile(
    r"^\s*(?:hola|buenas(?:\s+(?:d[ií]as|tardes|noches))?|buenos\s+(?:d[ií]as|tardes|noches)|"
    r"hey|hi\b|hello|qu[eé]\s+tal|c[oó]mo\s+est[aá]s?)\b",
    re.IGNORECASE,
)

_RESPUESTA_IDENTIDAD = (
    "Soy **lucIA**, asistente especializado en seguridad de la información.\n\n"
    "Trabajo con las normas ISO/IEC 27001 e ISO/IEC 27002 y puedo ayudarte con:\n\n"
    "- **Consulta normativa**: pregúntame sobre cualquier control, cláusula o requisito "
    "de las normas (p. ej. «¿qué dice el control 5.15?» o «explícame los controles de acceso»)\n"
    "- **Análisis de cumplimiento**: comparo tus documentos internos —políticas, procedimientos— "
    "con los requisitos de la norma para detectar brechas "
    "(p. ej. «¿cumple nuestra política de contraseñas con la ISO 27002?»)\n\n"
    "¿En qué puedo ayudarte?"
)

_RESPUESTA_SALUDO = (
    "Hola. Soy **lucIA**, asistente de seguridad de la información.\n\n"
    "Puedes preguntarme sobre la documentación interna, la normativa ISO 27001/27002 "
    "o pedirme un análisis de cumplimiento de un control concreto."
)


def respuesta_identidad(texto: str) -> str | None:
    t = texto.strip()
    if _PATRON_IDENTIDAD.match(t):
        return _RESPUESTA_IDENTIDAD
    if len(t) <= 40 and _PATRON_SALUDO.match(t):
        return _RESPUESTA_SALUDO
    return None


# ── Prompts de sistema por modo ──────────────────────────────────────────────

_SISTEMA_NORMA = (
    "Eres lucIA, asistente de seguridad de la información. Respondes SIEMPRE en español.\n\n"
    "RESTRICCIÓN: Los documentos incluidos son exclusivamente normativa ISO/IEC. "
    "Responde ÚNICAMENTE con la información que aparezca en esos documentos. "
    "No hagas referencia a documentación interna de ninguna empresa ni a ejemplos de implementación.\n\n"
    "FORMATO:\n"
    "• Empieza con 'X.Y — Título del control/cláusula' si hay referencia numérica.\n"
    "• Una frase de síntesis del propósito del control.\n"
    "• Lista de 3-5 puntos clave concretos que exige o recomienda la norma.\n"
    "• Al usar información de un documento, añade [N] al final de la frase "
    "(p. ej. '…el alcance debe estar documentado [1].'). "
    "Sin preguntas retóricas, sin frases de relleno.\n\n"
    "Si la información no está en los documentos: "
    "'Esta consulta no está cubierta por la normativa disponible.'"
)

_SISTEMA_INTERNA = (
    "Eres lucIA, asistente de seguridad de la información. Respondes SIEMPRE en español.\n\n"
    "RESTRICCIÓN: Los documentos incluidos son exclusivamente documentación interna de la organización. "
    "Responde ÚNICAMENTE con la información que aparezca en esos documentos. "
    "No hagas referencia a normativas ISO, estándares externos ni buenas prácticas del sector.\n\n"
    "REGLAS:\n"
    "1. Si los documentos definen un esquema con criterios, niveles o ejemplos "
    "(clasificación de la información, categorías de riesgo, tipos de incidente…), "
    "aplica esos criterios para dar una respuesta directa. Puedes razonar por "
    "eliminación: si un nivel excluye explícitamente un tipo de dato, ese dato "
    "pertenece a un nivel superior; si un nivel se describe como 'por defecto para "
    "lo no clasificado', los activos clasificados no pertenecen a él. "
    "Da la respuesta concreta y justifícala citando el criterio o ejemplo del "
    "documento que la soporta.\n"
    "2. Usa puntos concretos con los datos exactos que aparecen en los documentos.\n"
    "3. Si hay nombres propios, códigos o etiquetas específicas, cítalos literalmente.\n"
    "4. Al usar información de un documento, añade [N] al final de la frase. "
    "Sin preguntas retóricas, sin frases de relleno.\n\n"
    "Si la información no está en los documentos ni puede inferirse de sus criterios: "
    "'Esta información no figura en la documentación interna disponible.'"
)

_SISTEMA_COMPARACION = (
    "Eres lucIA, asistente de seguridad de la información. Respondes SIEMPRE en español.\n\n"
    "Los documentos están divididos en dos secciones claramente etiquetadas:\n"
    "  • NORMATIVA ISO: lo que exige o recomienda la norma ISO/IEC.\n"
    "  • DOCUMENTACIÓN INTERNA: lo que tiene implementado la organización.\n\n"
    "Tu tarea es un análisis de cumplimiento estructurado:\n"
    "1. Resume qué exige/recomienda la normativa sobre el tema consultado, "
    "citando ÚNICAMENTE lo que aparezca textualmente en los fragmentos [N…]. "
    "Si ningún fragmento [N] cubre el requisito concreto, indícalo explícitamente "
    "en lugar de usar conocimiento externo.\n"
    "2. Resume qué establece la documentación interna sobre ese mismo tema.\n"
    "3. Compara punto a punto de forma concreta y específica.\n"
    "4. Concluye con 'CUMPLIMIENTO: COMPLETO / PARCIAL / NO CUMPLE / SIN INFORMACIÓN' "
    "y una justificación breve.\n\n"
    "CRITERIOS DE EVALUACIÓN:\n"
    "- COMPLETO: la documentación interna implementa la sustancia del control "
    "(aunque use nombres o niveles distintos a los de la norma).\n"
    "- PARCIAL: implementa el concepto pero faltan elementos requeridos.\n"
    "- NO CUMPLE: el requisito está ausente en la documentación interna.\n"
    "- SIN INFORMACIÓN: el tema no aparece en los documentos disponibles; "
    "nunca uses NO CUMPLE por mera ausencia de datos.\n\n"
    "En el paso 2, cita datos concretos de la documentación interna (nombres propios, cifras, "
    "procedimientos específicos) que den respuesta al control. No uses frases genéricas.\n\n"
    "Al citar información de un documento, añade [N1], [N2], [I1], [I2], etc. al final de la frase. "
    "Sin preguntas retóricas, sin frases de relleno."
)

_SISTEMA_GENERAL = (
    "Eres lucIA, asistente de seguridad de la información. Respondes SIEMPRE en español.\n\n"
    "REGLAS:\n"
    "1. Usa exclusivamente los documentos numerados [1], [2]... del mensaje. "
    "No uses conocimiento externo.\n"
    "2. Si los documentos definen un esquema con criterios (niveles de clasificación, "
    "categorías de riesgo, tipos de incidente…), aplica esos criterios para dar una respuesta "
    "directa y concreta. No listes posibilidades — da la respuesta correcta y justifícala "
    "citando el criterio del documento que la soporta.\n"
    "3. Nunca añadas 'CUMPLIMIENTO: …' salvo que la pregunta use explícitamente palabras "
    "como 'cumple', 'cumplimiento', 'conforme' o 'se ajusta'.\n\n"
    "FORMATO:\n"
    "• Respuesta directa en 1-2 frases con el dato concreto.\n"
    "• Lista de 2-4 puntos de apoyo citando los documentos con datos exactos.\n"
    "• Al usar información de un documento, añade [N] al final de la frase. "
    "Sin preguntas retóricas, sin frases de relleno.\n\n"
    "Si la información no está en los documentos: "
    "'Esta consulta no está cubierta por los documentos disponibles.'"
)

_SISTEMAS = {
    "norma": _SISTEMA_NORMA,
    "interna": _SISTEMA_INTERNA,
    "comparacion": _SISTEMA_COMPARACION,
    "general": _SISTEMA_GENERAL,
}

# ── Prompts neutralizados para evaluación ciega ──────────────────────────────
# Conservan todas las restricciones de tarea (idioma, fuente única, criterios
# de veredicto). Solo eliminan el bloque de FORMATO que produce la firma visual.

_SISTEMA_NORMA_BRUTO = (
    "Eres lucIA, asistente de seguridad de la información. Respondes SIEMPRE en español.\n\n"
    "RESTRICCIÓN: Los documentos incluidos son exclusivamente normativa ISO/IEC. "
    "Responde ÚNICAMENTE con información de esos documentos. "
    "No hagas referencia a documentación interna de ninguna organización.\n\n"
    "FORMATO: Responde en prosa continua, sin encabezados, sin '§ X.Y — Título' "
    "y sin listas con viñetas. Si citas una cláusula o control, hazlo en el texto "
    "y solo si su numeración aparece literalmente en los documentos; nunca la inventes. "
    "Sin marcadores [N], sin frases de relleno.\n\n"
    "Si la información no está en los documentos: "
    "'Esta consulta no está cubierta por los documentos disponibles.'"
)

_SISTEMA_INTERNA_BRUTO = (
    "Eres lucIA, asistente de seguridad de la información. Respondes SIEMPRE en español.\n\n"
    "RESTRICCIÓN: Los documentos incluidos son exclusivamente documentación interna. "
    "Responde ÚNICAMENTE con información que aparezca en ellos. "
    "No menciones normativas ISO, estándares externos ni buenas prácticas del sector.\n\n"
    "FORMATO: Responde en prosa continua, sin encabezados y sin listas con viñetas. "
    "Si hay nombres propios, códigos o etiquetas específicas, cítalos literalmente. "
    "Sin marcadores [N], sin frases de relleno.\n\n"
    "Si la información no está en los documentos: "
    "'Esta información no figura en la documentación interna disponible.'"
)

_SISTEMA_COMPARACION_BRUTO = (
    "Eres lucIA, asistente de seguridad de la información. Respondes SIEMPRE en español.\n\n"
    "Los documentos están divididos en dos secciones: la normativa ISO/IEC (lo que exige "
    "o recomienda la norma) y la documentación interna (lo que tiene implementado la "
    "organización).\n\n"
    "Tu tarea es un análisis de cumplimiento. En prosa continua, sin encabezados, sin "
    "viñetas y sin pasos numerados: resume qué exige la normativa citando ÚNICAMENTE lo "
    "que aparezca en los fragmentos [N…] disponibles (si no aparece, indícalo en lugar "
    "de usar conocimiento externo), resume qué establece "
    "la documentación interna citando datos concretos (nombres propios, cifras, "
    "procedimientos), compara ambos y concluye con un veredicto expresado en lenguaje "
    "natural (se cumple, se cumple parcialmente, no se cumple, o no hay información "
    "suficiente) seguido de una justificación breve.\n\n"
    "CRITERIOS: cumplimiento completo si la documentación interna implementa la sustancia "
    "del control aunque use otros nombres; parcial si faltan elementos requeridos; "
    "incumplimiento si el requisito está ausente. No confundas ausencia de información "
    "con incumplimiento: si el tema no aparece, indícalo como falta de información.\n\n"
    "No uses la etiqueta 'CUMPLIMIENTO: ...'. Sin marcadores [N], sin frases de relleno."
)

_SISTEMA_GENERAL_BRUTO = (
    "RESTRICCIÓN ABSOLUTA: Eres un sistema RAG. Tu ÚNICA fuente de información son los "
    "documentos numerados incluidos en cada mensaje. Si un dato no aparece en ellos, no lo "
    "menciones. Usar conocimiento externo es un error grave.\n\n"
    "Eres lucIA, asistente de seguridad de la información. Respondes SIEMPRE en español.\n\n"
    "FORMATO: Responde en prosa continua, sin encabezados, sin '§ X.Y — Título' y sin "
    "listas con viñetas. Incluye un veredicto en lenguaje natural (se cumple / parcialmente / "
    "no se cumple / sin información) ÚNICAMENTE si la pregunta plantea de forma explícita una "
    "cuestión de cumplimiento (con palabras como 'cumple', 'cumplimiento', 'se ajusta', "
    "'cubre la norma'). Si es una consulta informativa, NO incluyas ningún veredicto ni frase "
    "de cierre evaluativa. Usa 'sin información' si el tema no aparece; nunca concluyas "
    "incumplimiento por mera ausencia de datos. Sin marcadores [N], sin relleno.\n\n"
    "Si la información no está en los documentos: "
    "'Esta consulta no está cubierta por los documentos disponibles.'"
)

_SISTEMAS_BRUTO = {
    "norma": _SISTEMA_NORMA_BRUTO,
    "interna": _SISTEMA_INTERNA_BRUTO,
    "comparacion": _SISTEMA_COMPARACION_BRUTO,
    "general": _SISTEMA_GENERAL_BRUTO,
}

_SISTEMA_EVIDENCIAS = (
    "Eres lucIA, asistente de seguridad de la información. Respondes SIEMPRE en español.\n\n"
    "Se te proporciona el texto de un reporte de evidencias operativas "
    "(escaneo de vulnerabilidades, resultados de herramientas como Tenable, Scorecard, Nessus u otras).\n\n"
    "TU TAREA:\n"
    "Enumera de forma concisa todos los hallazgos, vulnerabilidades o resultados relevantes del reporte.\n\n"
    "FORMATO PARA CADA HALLAZGO:\n"
    "N. **[ID/CVE si existe] Nombre o título** — Severidad: CRÍTICA/ALTA/MEDIA/BAJA/INFORMATIVA\n"
    "   Descripción: una o dos líneas indicando el problema y el recurso/sistema afectado.\n\n"
    "CIERRE CON UN RESUMEN:\n"
    "**Resumen:** X hallazgos totales — Crítica: N | Alta: N | Media: N | Baja: N | Informativa: N\n\n"
    "REGLAS:\n"
    "- Extrae los datos del reporte tal como aparecen; no inventes datos.\n"
    "- Si el reporte está en inglés, traduce los títulos al español.\n"
    "- Si la severidad no está indicada, escribe 'Sin clasificar'.\n"
    "- Sin frases de relleno ni introducción. Empieza directamente con el hallazgo 1.\n"
    "- Si se plantea una pregunta adicional, respóndela después del resumen."
)

_MAX_TOKENS_EVIDENCIAS = 2048
_MAX_TOKENS_POR_MODO = {"comparacion": 2500}

_MODO_LABEL = {
    "norma": "consulta normativa ISO",
    "interna": "consulta documentación interna",
    "comparacion": "análisis de cumplimiento",
    "general": "consulta general",
}


# ── Construcción del contexto ────────────────────────────────────────────────

def _resumir_seccion(seccion: str, max_len: int = 80) -> str:
    if len(seccion) <= max_len:
        return seccion
    corte = seccion[:max_len].rsplit(' ', 1)[0]
    return f"{corte}…"


def _construir_contexto(chunks: list[str], metas: list[dict], intencion: str) -> str:
    if intencion != "comparacion":
        partes = []
        for i, (chunk, meta) in enumerate(zip(chunks, metas), 1):
            tipo = _ETIQUETAS_TIPO.get(meta.get("tipo_doc", ""), "Documento")
            clausula = meta.get("clausula", "")
            seccion = meta.get("seccion", "")
            ref = f"Cláusula {clausula}" if clausula else (_resumir_seccion(seccion) if seccion else "")
            cabecera = f"[{i}] {tipo.upper()}" + (f" — {ref}" if ref else "")
            partes.append(f"{cabecera}\n{chunk}")
        return "\n\n".join(partes)

    # Modo comparación: separar normativa de documentación interna
    norma = [(c, m) for c, m in zip(chunks, metas) if m.get("tipo_doc") == "norma_iso"]
    interna = [(c, m) for c, m in zip(chunks, metas) if m.get("tipo_doc") != "norma_iso"]

    partes: list[str] = []

    if norma:
        partes.append("=== NORMATIVA ISO/IEC ===")
        for i, (chunk, meta) in enumerate(norma, 1):
            clausula = meta.get("clausula", "")
            seccion = meta.get("seccion", "")
            ref = f"Cláusula {clausula}" if clausula else (_resumir_seccion(seccion) if seccion else "")
            cabecera = f"[N{i}] NORMA ISO" + (f" — {ref}" if ref else "")
            partes.append(f"{cabecera}\n{chunk}")

    if interna:
        partes.append("\n=== DOCUMENTACIÓN INTERNA ===")
        for i, (chunk, meta) in enumerate(interna, 1):
            tipo = _ETIQUETAS_TIPO.get(meta.get("tipo_doc", ""), "Doc. interno")
            clausula = meta.get("clausula", "")
            seccion = meta.get("seccion", "")
            ref = f"Cláusula {clausula}" if clausula else (_resumir_seccion(seccion) if seccion else "")
            cabecera = f"[I{i}] {tipo.upper()}" + (f" — {ref}" if ref else "")
            partes.append(f"{cabecera}\n{chunk}")

    return "\n\n".join(partes)


def _construir_mensajes(
    chunks: list[str],
    metas: list[dict],
    pregunta: str,
    historial: list[dict],
    intencion: str,
    modo_salida: str = "estructurado",
) -> list[dict]:
    tabla = _SISTEMAS_BRUTO if modo_salida == "bruto" else _SISTEMAS
    sistema = tabla.get(intencion, tabla["general"])
    mensajes = [{"role": "system", "content": sistema}]

    for msg in historial[-MAX_TURNOS_HISTORIAL:]:
        if msg["rol"] in ("user", "assistant"):
            mensajes.append({"role": msg["rol"], "content": msg["contenido"]})

    if chunks:
        contexto = _construir_contexto(chunks, metas, intencion)
        if intencion == "comparacion":
            contenido_usuario = (
                f"Documentos de referencia:\n\n{contexto}\n\n"
                f"---\n\n"
                f"Pregunta: {pregunta}\n\n"
                f"Analiza el cumplimiento comparando la sección normativa con la documentación interna."
            )
        else:
            contenido_usuario = (
                f"Documentos de referencia (usa SOLO estos):\n\n{contexto}\n\n"
                f"---\n\n"
                f"Pregunta: {pregunta}\n\n"
                f"Recuerda: responde exclusivamente con la información de los documentos anteriores."
            )
    else:
        contenido_usuario = pregunta

    mensajes.append({"role": "user", "content": contenido_usuario})
    return mensajes


def generar_respuesta(
    chunks: list[str],
    metas: list[dict],
    pregunta: str,
    historial: list[dict] | None = None,
    intencion: str = "general",
    modo_salida: str = "estructurado",
) -> Iterator[str]:
    mensajes = _construir_mensajes(chunks, metas, pregunta, historial or [], intencion, modo_salida)
    max_tok = _MAX_TOKENS_POR_MODO.get(intencion, MAX_TOKENS)

    if MODO == "local":
        import ollama
        stream = ollama.chat(
            model=MODELO_LLM,
            messages=mensajes,
            stream=True,
            options={
                "temperature": TEMPERATURE,
                "num_predict": max_tok,
                "repeat_penalty": 1.3,
                "stop": _STOPS,
            },
        )
        for parte in stream:
            yield parte["message"]["content"]
    else:
        from groq import Groq
        stream = Groq(api_key=GROQ_API_KEY).chat.completions.create(
            model=MODELO_GROQ,
            messages=mensajes,
            stream=True,
            temperature=TEMPERATURE,
            max_tokens=max_tok,
            frequency_penalty=1.0,
            presence_penalty=0.3,
            stop=_STOPS,
        )
        for chunk in stream:
            yield chunk.choices[0].delta.content or ""


def analizar_evidencias(texto_reporte: str, consulta: str = "") -> Iterator[str]:
    MAX_CHARS = 40_000
    if len(texto_reporte) > MAX_CHARS:
        texto_reporte = (
            texto_reporte[:MAX_CHARS]
            + "\n\n[REPORTE TRUNCADO — se muestran los primeros 40 000 caracteres]"
        )

    contenido = f"REPORTE DE EVIDENCIAS:\n\n{texto_reporte}"
    if consulta.strip():
        contenido += f"\n\n---\nPregunta adicional: {consulta}"

    mensajes = [
        {"role": "system", "content": _SISTEMA_EVIDENCIAS},
        {"role": "user", "content": contenido},
    ]

    if MODO == "local":
        import ollama
        stream = ollama.chat(
            model=MODELO_LLM,
            messages=mensajes,
            stream=True,
            options={
                "temperature": TEMPERATURE,
                "num_predict": _MAX_TOKENS_EVIDENCIAS,
                "repeat_penalty": 1.3,
            },
        )
        for parte in stream:
            yield parte["message"]["content"]
    else:
        from groq import Groq
        stream = Groq(api_key=GROQ_API_KEY).chat.completions.create(
            model=MODELO_GROQ,
            messages=mensajes,
            stream=True,
            temperature=TEMPERATURE,
            max_tokens=_MAX_TOKENS_EVIDENCIAS,
            frequency_penalty=1.0,
            presence_penalty=0.3,
        )
        for chunk in stream:
            yield chunk.choices[0].delta.content or ""


def preguntar(coleccion: chromadb.Collection, pregunta: str) -> None:
    resp = respuesta_identidad(pregunta)
    if resp:
        console.print(f"\n[bold]Respuesta:[/bold] {resp}")
        console.print("[dim]" + "-" * 60 + "[/dim]")
        return

    if es_inyeccion_prompt(pregunta):
        console.print(
            "\n[bold]Respuesta:[/bold] Esta solicitud parece intentar modificar mi "
            "comportamiento o rol. Solo puedo responder preguntas sobre los documentos disponibles."
        )
        console.print("[dim]" + "-" * 60 + "[/dim]")
        return

    intencion = detectar_intencion(pregunta)
    console.print(f"[dim]Modo: {_MODO_LABEL[intencion]}[/dim]")

    chunks, metas = buscar_contexto(coleccion, pregunta, intencion=intencion)

    if not chunks:
        console.print("\n[bold]Respuesta:[/bold] No se encontraron fragmentos relevantes en la documentación.")
        console.print("[dim]" + "-" * 60 + "[/dim]")
        return

    respuesta_completa = ""
    for texto in generar_respuesta(chunks, metas, pregunta, intencion=intencion):
        respuesta_completa += texto
    respuesta_display = reemplazar_citas(respuesta_completa, metas)
    console.print(f"\n[bold]Respuesta:[/bold] {respuesta_display}")
    metas_citadas = extraer_metas_citadas(respuesta_completa, metas, intencion)
    console.print(f"\n[dim]Fuentes: {resumen_fuentes(metas_citadas)}[/dim]")
    console.print("[dim]" + "-" * 60 + "[/dim]")
