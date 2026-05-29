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
    r"|(muéstrame?|repite|dame?|revela|imprime|muestra)\s+(todo\s+)?(el\s+)?(contexto|prompt|instrucciones?)"
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


def resumen_fuentes(metas: list[dict]) -> str:
    por_doc: dict[str, list] = {}
    for meta in metas:
        fuente = meta["fuente"]
        tipo = _ETIQUETAS_TIPO.get(meta.get("tipo_doc", ""), "")
        clave = f"{fuente} [{tipo}]" if tipo else fuente
        por_doc.setdefault(clave, [])
        ref = meta.get("clausula") or meta.get("seccion", "")[:40]
        if ref and ref not in por_doc[clave]:
            por_doc[clave].append(ref)

    partes = []
    for clave, refs in por_doc.items():
        if refs:
            partes.append(f"{clave} ({', '.join(refs)})")
        else:
            partes.append(clave)
    return " | ".join(partes)


_STOPS = ["Pregunta:", "Usuario:", "\n¿", "\nAssistant:"]

_PATRON_IDENTIDAD = re.compile(
    r"^\s*(qu[eé]\s+eres|qui[eé]n\s+eres|qu[eé]\s+(puedes|haces|sabes|ofreces)|"
    r"c[oó]mo\s+te\s+llamas|cu[aá]l\s+es\s+tu\s+(nombre|funci[oó]n|prop[oó]sito)|"
    r"para\s+qu[eé]\s+sirves?|qu[eé]\s+tipo\s+de\s+asistente|"
    r"d[ií]me\s+qu[eé]\s+eres|"
    r"hola|buenos\s+(d[ií]as|tardes|noches)|buenas|hey|hi|hello|"
    r"qu[eé]\s+tal|c[oó]mo\s+est[aá]s?)[.!?,\s]*$",
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


def respuesta_identidad(texto: str) -> str | None:
    return _RESPUESTA_IDENTIDAD if _PATRON_IDENTIDAD.match(texto.strip()) else None


# ── Prompts de sistema por modo ──────────────────────────────────────────────

_SISTEMA_NORMA = (
    "Eres lucIA, asistente de seguridad de la información. Respondes SIEMPRE en español.\n\n"
    "RESTRICCIÓN: Los documentos incluidos son exclusivamente normativa ISO/IEC. "
    "Responde ÚNICAMENTE con la información que aparezca en esos documentos. "
    "No hagas referencia a documentación interna de ninguna empresa ni a ejemplos de implementación.\n\n"
    "FORMATO:\n"
    "• Empieza con '§ X.Y — Título del control/cláusula' si hay referencia numérica.\n"
    "• Una frase de síntesis del propósito del control.\n"
    "• Lista de 3-5 puntos clave concretos que exige o recomienda la norma.\n"
    "• Sin marcadores [N], sin preguntas retóricas, sin frases de relleno.\n\n"
    "Si la información no está en los documentos: "
    "'Esta consulta no está cubierta por la normativa disponible.'"
)

_SISTEMA_INTERNA = (
    "Eres lucIA, asistente de seguridad de la información. Respondes SIEMPRE en español.\n\n"
    "RESTRICCIÓN: Los documentos incluidos son exclusivamente documentación interna de la organización. "
    "Responde ÚNICAMENTE con la información que aparezca en esos documentos. "
    "No hagas referencia a normativas ISO, estándares externos ni buenas prácticas del sector.\n\n"
    "FORMATO:\n"
    "• Describe lo que establece la documentación interna sobre el tema consultado.\n"
    "• Usa puntos concretos con los datos exactos que aparecen en los documentos.\n"
    "• Si hay nombres propios, códigos o etiquetas específicas, cítalos literalmente.\n"
    "• Sin marcadores [N], sin preguntas retóricas, sin frases de relleno.\n\n"
    "Si la información no está en los documentos: "
    "'Esta información no figura en la documentación interna disponible.'"
)

_SISTEMA_COMPARACION = (
    "Eres lucIA, asistente de seguridad de la información. Respondes SIEMPRE en español.\n\n"
    "Los documentos están divididos en dos secciones claramente etiquetadas:\n"
    "  • NORMATIVA ISO: lo que exige o recomienda la norma ISO/IEC.\n"
    "  • DOCUMENTACIÓN INTERNA: lo que tiene implementado la organización.\n\n"
    "Tu tarea es un análisis de cumplimiento estructurado:\n"
    "1. Resume qué exige/recomienda la normativa sobre el tema consultado.\n"
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
    "Sin marcadores [N], sin preguntas retóricas, sin frases de relleno."
)

_SISTEMA_GENERAL = (
    "RESTRICCIÓN ABSOLUTA: Eres un sistema RAG. Tu ÚNICA fuente de información son los "
    "documentos numerados [1], [2]... incluidos en cada mensaje. "
    "Si un dato no aparece en esos documentos, no lo menciones. "
    "Usar conocimiento externo es un error grave.\n\n"
    "Eres lucIA, asistente de seguridad de la información. Respondes SIEMPRE en español.\n\n"
    "FORMATO DE RESPUESTA:\n"
    "• Consulta informativa o normativa: empieza con '§ X.Y — Título' solo si esa numeración "
    "aparece literalmente en los documentos (nunca inventes números de cláusula); "
    "una frase de síntesis y una lista de 3-4 puntos clave concretos. "
    "NO incluyas la línea 'CUMPLIMIENTO: ...' en respuestas puramente informativas.\n"
    "• Análisis de cumplimiento (solo cuando la pregunta contenga palabras como 'cumple', "
    "'cumplimiento', 'se ajusta', 'cubre la norma' o similares): contexto breve, "
    "aspectos cubiertos/no cubiertos, y cierra con "
    "'CUMPLIMIENTO: COMPLETO / PARCIAL / NO CUMPLE / SIN INFORMACIÓN'.\n"
    "  - Usa SIN INFORMACIÓN si el tema no aparece en los documentos; "
    "nunca uses NO CUMPLE por mera ausencia de datos.\n"
    "• Interpreta y sintetiza en español; no traduzcas párrafos literales.\n"
    "• Sin marcadores [N], sin preguntas, sin frases de relleno.\n\n"
    "Si la información no está en los documentos: "
    "'Esta consulta no está cubierta por los documentos disponibles.'"
)

_SISTEMAS = {
    "norma": _SISTEMA_NORMA,
    "interna": _SISTEMA_INTERNA,
    "comparacion": _SISTEMA_COMPARACION,
    "general": _SISTEMA_GENERAL,
}

_MODO_LABEL = {
    "norma": "consulta normativa ISO",
    "interna": "consulta documentación interna",
    "comparacion": "análisis de cumplimiento",
    "general": "consulta general",
}


# ── Construcción del contexto ────────────────────────────────────────────────

def _construir_contexto(chunks: list[str], metas: list[dict], intencion: str) -> str:
    if intencion != "comparacion":
        partes = []
        for i, (chunk, meta) in enumerate(zip(chunks, metas), 1):
            tipo = _ETIQUETAS_TIPO.get(meta.get("tipo_doc", ""), "Documento")
            clausula = meta.get("clausula", "")
            seccion = meta.get("seccion", "")
            ref = f"Cláusula {clausula}" if clausula else (seccion[:50] if seccion else "")
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
            ref = f"Cláusula {clausula}" if clausula else (seccion[:50] if seccion else "")
            cabecera = f"[N{i}] NORMA ISO" + (f" — {ref}" if ref else "")
            partes.append(f"{cabecera}\n{chunk}")

    if interna:
        partes.append("\n=== DOCUMENTACIÓN INTERNA ===")
        for i, (chunk, meta) in enumerate(interna, 1):
            tipo = _ETIQUETAS_TIPO.get(meta.get("tipo_doc", ""), "Doc. interno")
            clausula = meta.get("clausula", "")
            seccion = meta.get("seccion", "")
            ref = f"Cláusula {clausula}" if clausula else (seccion[:50] if seccion else "")
            cabecera = f"[I{i}] {tipo.upper()}" + (f" — {ref}" if ref else "")
            partes.append(f"{cabecera}\n{chunk}")

    return "\n\n".join(partes)


def _construir_mensajes(
    chunks: list[str],
    metas: list[dict],
    pregunta: str,
    historial: list[dict],
    intencion: str,
) -> list[dict]:
    sistema = _SISTEMAS.get(intencion, _SISTEMA_GENERAL)
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
) -> Iterator[str]:
    mensajes = _construir_mensajes(chunks, metas, pregunta, historial or [], intencion)

    if MODO == "local":
        import ollama
        stream = ollama.chat(
            model=MODELO_LLM,
            messages=mensajes,
            stream=True,
            options={
                "temperature": TEMPERATURE,
                "num_predict": MAX_TOKENS,
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
            max_tokens=MAX_TOKENS,
            frequency_penalty=1.0,
            presence_penalty=0.3,
            stop=_STOPS,
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

    console.print("\n[bold]Respuesta:[/bold] ", end="")
    for texto in generar_respuesta(chunks, metas, pregunta, intencion=intencion):
        print(texto, end="", flush=True)
    print()
    console.print(f"\n[dim]Fuentes: {resumen_fuentes(metas)}[/dim]")
    console.print("[dim]" + "-" * 60 + "[/dim]")
