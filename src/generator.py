import re
from typing import Iterator

import chromadb

from .config import (
    GROQ_API_KEY, MAX_TOKENS, MAX_TURNOS_HISTORIAL,
    MODO, MODELO_GROQ, MODELO_LLM, TEMPERATURE, console,
)
from .retriever import buscar_contexto

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
    r"|jailbreak|\\bDAN\\b"
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


def _construir_contexto(chunks: list[str], metas: list[dict]) -> str:
    partes = []
    for i, (chunk, meta) in enumerate(zip(chunks, metas), 1):
        tipo = _ETIQUETAS_TIPO.get(meta.get("tipo_doc", ""), "Documento")
        clausula = meta.get("clausula", "")
        seccion = meta.get("seccion", "")
        ref = f"Cláusula {clausula}" if clausula else (seccion[:50] if seccion else "")
        cabecera = f"[{i}] {tipo.upper()}" + (f" — {ref}" if ref else "")
        partes.append(f"{cabecera}\n{chunk}")
    return "\n\n".join(partes)


_SISTEMA = (
    "RESTRICCIÓN ABSOLUTA: Eres un sistema RAG. Tu ÚNICA fuente de información son los "
    "documentos numerados [1], [2]... incluidos en cada mensaje. "
    "Si un dato no aparece en esos documentos, no lo menciones. "
    "Usar conocimiento externo es un error grave.\n\n"

    "Eres lucIA, asistente de seguridad de la información. Respondes SIEMPRE en español.\n\n"

    "FORMATO DE RESPUESTA:\n"
    "• Consulta normativa: empieza con '§ X.Y — Título' si hay cláusula, "
    "una frase de síntesis del propósito, y una lista de 3-4 puntos clave concretos.\n"
    "• Análisis de cumplimiento: contexto breve, lista de aspectos cubiertos/no cubiertos, "
    "y cierra con 'CUMPLIMIENTO: COMPLETO / PARCIAL / NO CUMPLE'.\n"
    "• Interpreta y sintetiza en español; no traduzcas párrafos literales.\n"
    "• Sin marcadores [N], sin preguntas, sin frases de relleno.\n\n"

    "Si la información no está en los documentos: "
    "'Esta consulta no está cubierta por los documentos disponibles.'"
)


def _construir_mensajes(
    chunks: list[str],
    metas: list[dict],
    pregunta: str,
    historial: list[dict],
) -> list[dict]:
    mensajes = [{"role": "system", "content": _SISTEMA}]

    for msg in historial[-MAX_TURNOS_HISTORIAL:]:
        if msg["rol"] in ("user", "assistant"):
            mensajes.append({"role": msg["rol"], "content": msg["contenido"]})

    if chunks:
        contexto = _construir_contexto(chunks, metas)
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
) -> Iterator[str]:
    mensajes = _construir_mensajes(chunks, metas, pregunta, historial or [])

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

    chunks, metas = buscar_contexto(coleccion, pregunta)

    if not chunks:
        console.print("\n[bold]Respuesta:[/bold] No se encontraron fragmentos relevantes en la documentación.")
        console.print("[dim]" + "-" * 60 + "[/dim]")
        return

    console.print("\n[bold]Respuesta:[/bold] ", end="")
    for texto in generar_respuesta(chunks, metas, pregunta):
        print(texto, end="", flush=True)
    print()
    console.print(f"\n[dim]Fuentes: {resumen_fuentes(metas)}[/dim]")
    console.print("[dim]" + "-" * 60 + "[/dim]")
