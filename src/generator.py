from typing import Iterator

import chromadb

from .config import GROQ_API_KEY, MAX_TURNOS_HISTORIAL, MODO, MODELO_GROQ, MODELO_LLM, console
from .retriever import buscar_contexto


def _etiqueta_ubicacion(meta: dict) -> str:
    fuente = meta["fuente"]
    if "pagina" in meta:
        return f"{fuente} · pág. {meta['pagina']}"
    if "seccion" in meta:
        return f"{fuente} · § {meta['seccion']}"
    return fuente


def resumen_fuentes(metas: list[dict]) -> str:
    por_doc: dict[str, list] = {}
    for meta in metas:
        fuente = meta["fuente"]
        por_doc.setdefault(fuente, [])
        if "pagina" in meta and meta["pagina"] not in por_doc[fuente]:
            por_doc[fuente].append(meta["pagina"])

    partes = []
    for fuente, paginas in por_doc.items():
        if paginas:
            partes.append(f"{fuente} (págs. {', '.join(str(p) for p in sorted(paginas))})")
        else:
            partes.append(fuente)
    return ", ".join(partes)


def _construir_mensajes(
    chunks: list[str],
    metas: list[dict],
    pregunta: str,
    historial: list[dict],
) -> list[dict]:
    contexto = "\n---\n".join(
        f"[{_etiqueta_ubicacion(meta)}]\n{chunk}"
        for chunk, meta in zip(chunks, metas)
    )
    sistema = (
        "Eres un asistente experto. Responde ÚNICAMENTE con la información del contexto proporcionado.\n"
        "Cita los documentos y páginas o secciones de los que extraes la información.\n"
        "Si la respuesta no está en el contexto, di exactamente:\n"
        '"No dispongo de información suficiente en la documentación para responder esta consulta."\n\n'
        f"CONTEXTO:\n{contexto}"
    )

    mensajes = [{"role": "system", "content": sistema}]

    # Incluir los últimos N turnos del historial (pares user/assistant)
    for msg in historial[-MAX_TURNOS_HISTORIAL:]:
        if msg["rol"] in ("user", "assistant"):
            mensajes.append({"role": msg["rol"], "content": msg["contenido"]})

    mensajes.append({"role": "user", "content": pregunta})
    return mensajes


def generar_respuesta(
    chunks: list[str],
    metas: list[dict],
    pregunta: str,
    historial: list[dict] | None = None,
) -> Iterator[str]:
    """Genera la respuesta del LLM en streaming con memoria conversacional."""
    mensajes = _construir_mensajes(chunks, metas, pregunta, historial or [])

    if MODO == "local":
        import ollama
        stream = ollama.chat(model=MODELO_LLM, messages=mensajes, stream=True)
        for parte in stream:
            yield parte["message"]["content"]
    else:
        from groq import Groq
        stream = Groq(api_key=GROQ_API_KEY).chat.completions.create(
            model=MODELO_GROQ,
            messages=mensajes,
            stream=True,
        )
        for chunk in stream:
            yield chunk.choices[0].delta.content or ""


def preguntar(coleccion: chromadb.Collection, pregunta: str) -> None:
    """CLI: busca contexto y responde con streaming al terminal (sin historial)."""
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
