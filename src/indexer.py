import hashlib
import json
import os
import re

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


# Headings: markdown, numeración ISO (5.1, 5.1.1, A.1), numeración española (1. Objeto)
_HEADING_RE = re.compile(
    r"^("
    r"#{1,6}\s+\S.+"
    r"|(?:[A-Z]\.)?(?:\d+\.)+\d*\.?\s+[A-ZÁÉÍÓÚÜÑa-záéíóúüñ].{0,120}"
    r"|\d+\.\s+[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñA-ZÁÉÍÓÚÜÑ].{0,80}"
    r")$",
    re.MULTILINE,
)

_CLAUSULA_RE = re.compile(r"^(?:[A-Z]\.)?(\d+(?:\.\d+)*)")

# Línea que es solo número de sección (p.ej. "5.27.1" o "A.2")
_LINEA_SOLO_NUM_RE = re.compile(r"^[A-Z]?\.?\d[\d\.]*\s*$")

# Frases de boilerplate legal/copyright en normas ISO
_BOILERPLATE = (
    "ISO copyright",
    "without permission in writing",
    "All rights reserved",
    "reproduced or utilized",
    "Ch. de Blandonnet",
    "CP 401",
    "Vernier, Geneva",
    "www.iso.org",
)


def _es_chunk_valido(chunk: str) -> bool:
    """Descarta chunks de tabla de contenidos, copyright y listas de números."""
    lineas = [l for l in chunk.splitlines() if l.strip()]
    if not lineas:
        return False
    # Boilerplate legal
    if any(trigger in chunk for trigger in _BOILERPLATE):
        return False
    # Tabla de contenidos: >45% de líneas son solo números/cláusulas
    n_toc = sum(1 for l in lineas if _LINEA_SOLO_NUM_RE.match(l.strip()))
    if len(lineas) >= 5 and n_toc / len(lineas) > 0.45:
        return False
    # Líneas muy cortas en promedio (índice, TOC esparso)
    avg = sum(len(l) for l in lineas) / len(lineas)
    if avg < 12 and len(lineas) > 8:
        return False
    # Chunk que es solo un título/heading sin cuerpo (≤ 2 líneas, < 120 chars total)
    if len(lineas) <= 2 and len(chunk.strip()) < 120:
        return False
    return True


def _limpiar_titulo(titulo: str) -> str:
    return re.sub(r"^#+\s+", "", titulo).strip()


def _tipo_documento(archivo: str) -> str:
    n = archivo.lower()
    if "27001" in n or "27002" in n:
        return "norma_iso"
    if re.match(r"pol[_\-]", n) or "politic" in n:
        return "politica"
    if re.match(r"proc[_\-]", n) or "procedimiento" in n:
        return "procedimiento"
    return "documento_interno"


def _extraer_clausula(heading: str) -> str | None:
    m = _CLAUSULA_RE.match(heading.strip())
    return m.group(1) if m else None


def _chunksificar_por_secciones(texto: str, archivo: str, tipo_doc: str) -> list[tuple[str, dict]]:
    matches = list(_HEADING_RE.finditer(texto))
    secciones: list[tuple[str | None, str]] = []

    if not matches:
        secciones.append((None, texto))
    else:
        if matches[0].start() > 0:
            secciones.append((None, texto[: matches[0].start()]))
        for i, m in enumerate(matches):
            fin = matches[i + 1].start() if i + 1 < len(matches) else len(texto)
            secciones.append((_limpiar_titulo(m.group(1)), texto[m.end(): fin]))

    resultado = []
    for heading, contenido in secciones:
        for chunk in splitter.split_text(contenido):
            if len(chunk.strip()) >= MIN_CHUNK_LEN and _es_chunk_valido(chunk):
                meta: dict = {"fuente": archivo, "tipo_doc": tipo_doc}
                if heading:
                    meta["seccion"] = heading
                    clausula = _extraer_clausula(heading)
                    if clausula:
                        meta["clausula"] = clausula
                resultado.append((chunk, meta))
    return resultado


def _extraer_chunks(ruta: str, archivo: str) -> list[tuple[str, dict]]:
    tipo_doc = _tipo_documento(archivo)
    texto = md_converter.convert(ruta).text_content
    # Unir palabras partidas por guión al final de línea (artefacto PDF)
    texto = re.sub(r"(\w)-\n(\w)", r"\1\2", texto)
    return _chunksificar_por_secciones(texto, archivo, tipo_doc)


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


def _listar_documentos() -> list[tuple[str, str]]:
    """Devuelve lista de (ruta_absoluta, nombre_relativo) para todos los docs en CARPETA_DOCS."""
    resultado = []
    for raiz, _, ficheros in os.walk(CARPETA_DOCS):
        for f in ficheros:
            if f.lower().endswith(EXTENSIONES):
                ruta_abs = os.path.join(raiz, f)
                rel = os.path.relpath(ruta_abs, CARPETA_DOCS).replace("\\", "/")
                resultado.append((ruta_abs, rel))
    return resultado


def indexar_documentos(coleccion: chromadb.Collection) -> None:
    documentos = _listar_documentos()

    if not documentos:
        console.print("[red]No hay documentos en 'documentos/'[/red]")
        return

    hashes_previos = _cargar_hashes()
    hashes_nuevos = {}
    cambios = False

    for ruta, archivo in documentos:
        hash_actual = _hash_archivo(ruta)
        hashes_nuevos[archivo] = hash_actual

        if hashes_previos.get(archivo) == hash_actual:
            console.print(f"   [dim]Sin cambios: {archivo}[/dim]")
            continue

        console.print(f"   [cyan]Indexando: {archivo}[/cyan]")
        try:
            chunks_con_meta = _extraer_chunks(ruta, archivo)

            try:
                existentes = coleccion.get(where={"fuente": archivo})
                if existentes["ids"]:
                    coleccion.delete(ids=existentes["ids"])
            except Exception:
                pass

            if chunks_con_meta:
                docs, metas = zip(*chunks_con_meta)
                id_base = re.sub(r"[^a-zA-Z0-9_\-]", "_", archivo)
                coleccion.add(
                    documents=list(docs),
                    metadatas=list(metas),
                    ids=[f"{id_base}_{i}" for i in range(len(docs))],
                )
            cambios = True
            console.print(f"      [green]→ {len(chunks_con_meta)} fragmentos[/green]")

        except Exception as e:
            console.print(f"      [red]Error procesando {archivo}: {e}[/red]")

    if cambios or hashes_nuevos != hashes_previos:
        _guardar_hashes(hashes_nuevos)
