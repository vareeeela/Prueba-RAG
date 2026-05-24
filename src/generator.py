import chromadb

from .config import GROQ_API_KEY, MODO, MODELO_GROQ, MODELO_LLM, console
from .retriever import buscar_contexto


def preguntar(coleccion: chromadb.Collection, pregunta: str) -> None:
    chunks, fuentes = buscar_contexto(coleccion, pregunta)
    docs_unicos = sorted(set(fuentes))

    contexto = "\n---\n".join(
        f"[Fragmento {i + 1} · {fuente}]\n{chunk}"
        for i, (chunk, fuente) in enumerate(zip(chunks, fuentes))
    )

    prompt = f"""Eres un asistente experto. Responde ÚNICAMENTE con la información del contexto proporcionado.
Cita los documentos de los que extraes la información.
Si la respuesta no está en el contexto, di exactamente:
"No dispongo de información suficiente en la documentación para responder esta consulta."

CONTEXTO (fuentes: {', '.join(docs_unicos)}):
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
    console.print(f"\n[dim]Fuentes: {', '.join(docs_unicos)}[/dim]")
    console.print("[dim]" + "-" * 60 + "[/dim]")
