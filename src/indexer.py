import hashlib
import json
import os
import re

import chromadb
import pdfplumber

from .config import (
    CACHE_HASHES, CARPETA_DOCS, COLECCION, EXTENSIONES,
    MIN_CHUNK_LEN, MODELO_EMBEDDINGS, RUTA_BD,
    console, embedding_fn, md_converter, splitter,
)


def _hash_archivo(ruta: str) -> str:
    h = hashlib.md5()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(65536), b""):
            h.update(bloque)
    return h.hexdigest()


def _cargar_hashes() -> dict:
    if os.path.exists(CACHE_HASHES):
        with open(CACHE_HASHES) as f:
            return json.load(f)
    return {}


def _guardar_hashes(hashes: dict) -> None:
    os.makedirs(RUTA_BD, exist_ok=True)
    with open(CACHE_HASHES, "w") as f:
        json.dump(hashes, f)


def _extraer_chunks_pdf(ruta: str, archivo: str) -> list[tuple[str, dict]]:
    """Extrae chunks de un PDF con número de página en los metadatos."""
    resultado = []
    with pdfplumber.open(ruta) as pdf:
        for num_pagina, pagina in enumerate(pdf.pages, start=1):
            texto = pagina.extract_text() or ""
            if not texto.strip():
                continue
            for chunk in splitter.split_text(texto):
                if len(chunk.strip()) >= MIN_CHUNK_LEN:
                    resultado.append((chunk, {"fuente": archivo, "pagina": num_pagina}))
    return resultado


def _extraer_chunks_texto(ruta: str, archivo: str) -> list[tuple[str, dict]]:
    """Extrae chunks de formatos no-PDF con sección más cercana en los metadatos."""
    texto = md_converter.convert(ruta).text_content
    heading_re = re.compile(r"^#{1,6}\s+(.+)", re.MULTILINE)

    # Dividir el texto en secciones según los encabezados
    matches = list(heading_re.finditer(texto))
    secciones: list[tuple[str | None, str]] = []

    if not matches:
        secciones.append((None, texto))
    else:
        if matches[0].start() > 0:
            secciones.append((None, texto[: matches[0].start()]))
        for i, m in enumerate(matches):
            fin = matches[i + 1].start() if i + 1 < len(matches) else len(texto)
            secciones.append((m.group(1).strip(), texto[m.end(): fin]))

    resultado = []
    for heading, contenido in secciones:
        for chunk in splitter.split_text(contenido):
            if len(chunk.strip()) >= MIN_CHUNK_LEN:
                meta: dict = {"fuente": archivo}
                if heading:
                    meta["seccion"] = heading
                resultado.append((chunk, meta))
    return resultado


def obtener_coleccion(cliente: chromadb.PersistentClient) -> chromadb.Collection:
    nombres = [c.name for c in cliente.list_collections()]
    if COLECCION in nombres:
        # Leer metadatos sin pasar EF para evitar el conflicto de validación de ChromaDB
        try:
            col_meta = cliente.get_collection(name=COLECCION)
            modelo_guardado = col_meta.metadata.get("embedding_model")
        except Exception:
            modelo_guardado = None

        if modelo_guardado != MODELO_EMBEDDINGS:
            console.print("[yellow]Modelo de embeddings actualizado — reindexando...[/yellow]")
            cliente.delete_collection(COLECCION)
            if os.path.exists(CACHE_HASHES):
                os.remove(CACHE_HASHES)
        else:
            return cliente.get_collection(name=COLECCION, embedding_function=embedding_fn)
    return cliente.create_collection(
        name=COLECCION,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine", "embedding_model": MODELO_EMBEDDINGS},
    )


def indexar_documentos(coleccion: chromadb.Collection) -> None:
    archivos = [f for f in os.listdir(CARPETA_DOCS) if f.lower().endswith(EXTENSIONES)]

    if not archivos:
        console.print("[red]No hay documentos en 'documentos/'[/red]")
        return

    hashes_previos = _cargar_hashes()
    hashes_nuevos = {}
    cambios = False

    for archivo in archivos:
        ruta = os.path.join(CARPETA_DOCS, archivo)
        hash_actual = _hash_archivo(ruta)
        hashes_nuevos[archivo] = hash_actual

        if hashes_previos.get(archivo) == hash_actual:
            console.print(f"   [dim]Sin cambios: {archivo}[/dim]")
            continue

        console.print(f"   [cyan]Indexando: {archivo}[/cyan]")
        try:
            if archivo.lower().endswith(".pdf"):
                chunks_con_meta = _extraer_chunks_pdf(ruta, archivo)
            else:
                chunks_con_meta = _extraer_chunks_texto(ruta, archivo)

            try:
                existentes = coleccion.get(where={"fuente": archivo})
                if existentes["ids"]:
                    coleccion.delete(ids=existentes["ids"])
            except Exception:
                pass

            if chunks_con_meta:
                docs, metas = zip(*chunks_con_meta)
                coleccion.add(
                    documents=list(docs),
                    metadatas=list(metas),
                    ids=[f"{archivo}_{i}" for i in range(len(docs))],
                )
            cambios = True
            console.print(f"      [green]→ {len(chunks_con_meta)} fragmentos[/green]")

        except Exception as e:
            console.print(f"      [red]Error procesando {archivo}: {e}[/red]")

    if cambios or hashes_nuevos != hashes_previos:
        _guardar_hashes(hashes_nuevos)
