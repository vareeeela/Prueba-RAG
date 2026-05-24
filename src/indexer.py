import hashlib
import json
import os

import chromadb

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


def obtener_coleccion(cliente: chromadb.PersistentClient) -> chromadb.Collection:
    nombres = [c.name for c in cliente.list_collections()]
    if COLECCION in nombres:
        col = cliente.get_collection(name=COLECCION, embedding_function=embedding_fn)
        if col.metadata.get("embedding_model") != MODELO_EMBEDDINGS:
            console.print("[yellow]Modelo de embeddings actualizado — reindexando...[/yellow]")
            cliente.delete_collection(COLECCION)
            if os.path.exists(CACHE_HASHES):
                os.remove(CACHE_HASHES)
        else:
            return col
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
            texto = md_converter.convert(ruta).text_content
            chunks = [c for c in splitter.split_text(texto) if len(c.strip()) > MIN_CHUNK_LEN]

            try:
                existentes = coleccion.get(where={"fuente": archivo})
                if existentes["ids"]:
                    coleccion.delete(ids=existentes["ids"])
            except Exception:
                pass

            coleccion.add(
                documents=chunks,
                metadatas=[{"fuente": archivo}] * len(chunks),
                ids=[f"{archivo}_{i}" for i in range(len(chunks))],
            )
            cambios = True
            console.print(f"      [green]→ {len(chunks)} fragmentos[/green]")

        except Exception as e:
            console.print(f"      [red]Error procesando {archivo}: {e}[/red]")

    if cambios or hashes_nuevos != hashes_previos:
        _guardar_hashes(hashes_nuevos)
