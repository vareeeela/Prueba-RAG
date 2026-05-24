import chromadb

from .config import GROQ_API_KEY, MODO, MODELO_GROQ, MODELO_LLM, console
from .retriever import buscar_contexto


def _etiqueta_ubicacion(meta: dict) -> str:
    fuente = meta["fuente"]
    if "pagina" in meta:
        return f"{fuente} · pág. {meta['pagina']}"
    if "seccion" in meta:
        return f"{fuente} · § {meta['seccion']}"
    return fuente


def _resumen_fuentes(metas: list[dict]) -> str:
    """Agrupa páginas por documento para el pie de respuesta."""
    por_doc: dict[str, list] = {}
    for meta in metas:
        fuente = meta["fuente"]
        por_doc.setdefault(fuente, [])
        if "pagina" in meta and meta["pagina"] not in por_doc[fuente]:
            por_doc[fuente].append(meta["pagina"])

    partes = []
    for fuente, paginas in por_doc.items():
        if paginas:
            paginas_str = ", ".join(str(p) for p in sorted(paginas))
            partes.append(f"{fuente} (págs. {paginas_str})")
        else:
            partes.append(fuente)
    return ", ".join(partes)


def preguntar(coleccion: chromadb.Collection, pregunta: str) -> None:
    chunks, metas = buscar_contexto(coleccion, pregunta)

    if not chunks:
        console.print("\n[bold]Respuesta:[/bold] No se encontraron fragmentos relevantes en la documentación.")
        console.print("[dim]" + "-" * 60 + "[/dim]")
        return

    contexto = "\n---\n".join(
        f"[{_etiqueta_ubicacion(meta)}]\n{chunk}"
        for chunk, meta in zip(chunks, metas)
    )

    prompt = f"""Eres un asistente experto. Responde ÚNICAMENTE con la información del contexto proporcionado.
Cita los documentos y páginas o secciones de los que extraes la información.
Si la respuesta no está en el contexto, di exactamente:
"No dispongo de información suficiente en la documentación para responder esta consulta."

CONTEXTO:
{contexto}

PREGUNTA: {pregunta}

RESPUESTA:"""

    console.print("\n[bold]Respuesta:[/bold] ", end="")

    if MODO == "local":
        import ollama
        stream = ollama.chat(
            model=MODELO_LLM,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for parte in stream:
            print(parte["message"]["content"], end="", flush=True)
    else:
        from groq import Groq
        cliente_groq = Groq(api_key=GROQ_API_KEY)
        stream = cliente_groq.chat.completions.create(
            model=MODELO_GROQ,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in stream:
            print(chunk.choices[0].delta.content or "", end="", flush=True)

    print()
    console.print(f"\n[dim]Fuentes: {_resumen_fuentes(metas)}[/dim]")
    console.print("[dim]" + "-" * 60 + "[/dim]")
