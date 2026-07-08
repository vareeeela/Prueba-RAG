"""
experimento.py — Runner del experimento de validación ciega de lucIA.

Importa la batería de preguntas de bateria.py y ejecuta cada una en dos modos
(estructurado / bruto), volcando los resultados a CSV.

Lógica por tipo:
  C1, C2, A → plano ciego: dos LLM calls (mismo contexto, distinto prompt)
  C3        → plano ambos: ídem; el evaluador usa bruto para contenido y
               estructurado para comprobar el veredicto CUMPLIMIENTO
  C4        → plano ciego + hilo: los cuatro turnos se ejecutan en orden con
               historial acumulado; se mantienen dos hilos independientes
               (uno por modo) para no contaminar la comparación
  B         → plano objetivo (honestidad): se ejecuta en ambos modos igual
               que el resto; el evaluador anota si declara fuera de alcance
  D         → plano objetivo (robustez): se comprueba es_inyeccion_prompt
               antes de llamar al LLM; si actúa el guardarraíl, se registra
               bloqueado=True y no se hace ninguna llamada al modelo

Columnas del CSV:
  id, categoria, plano, texto,
  intencion_detectada, intencion_esperada, bloqueado,
  respuesta_estructurada, fuentes_estructuradas,
  respuesta_bruta, fuentes_brutas,
  trazador, trazador_presente, nota

Uso:
    python experimento.py [salida.csv]
"""
import csv
import sys
import time

import chromadb
from groq import RateLimitError

from src.config import RUTA_BD
from src.generator import (
    es_inyeccion_prompt,
    generar_respuesta,
    resumen_fuentes,
    respuesta_identidad,
)
from src.indexer import indexar_documentos, obtener_coleccion
from src.retriever import buscar_contexto, detectar_intencion
from bateria import PREGUNTAS

SALIDA_CSV = "resultados_experimento.csv"

PAUSA_ENTRE_MODOS = 3      # segundos entre la llamada estructurada y la bruta
PAUSA_ENTRE_PREGUNTAS = 5  # segundos entre preguntas (rate limit Groq)

_CAMPOS_CSV = [
    "id", "categoria", "plano", "texto",
    "intencion_detectada", "intencion_esperada", "bloqueado",
    "respuesta_estructurada", "fuentes_estructuradas",
    "respuesta_bruta", "fuentes_brutas",
    "trazador", "trazador_presente", "nota",
]

_MSG_SIN_COBERTURA = "Esta consulta no está cubierta por los documentos disponibles."
_MSG_INYECCION = (
    "Esta solicitud parece intentar modificar mi comportamiento o rol. "
    "Solo puedo responder preguntas sobre los documentos disponibles."
)


def _recoger(gen) -> str:
    return "".join(gen)


def _procesar(
    coleccion,
    pregunta_dict: dict,
    hist_est: list,
    hist_bruto: list,
) -> dict:
    """
    Ejecuta una pregunta y devuelve el dict de resultados para el CSV.
    hist_est / hist_bruto son los historiales acumulados del hilo C4
    (listas vacías para preguntas independientes).
    """
    texto = pregunta_dict["texto"]
    plano = pregunta_dict.get("plano", "ciego")
    trazador = pregunta_dict.get("trazador")

    fila: dict = {
        "id": pregunta_dict["id"],
        "categoria": pregunta_dict["categoria"],
        "plano": plano,
        "texto": texto,
        "intencion_detectada": "",
        "intencion_esperada": pregunta_dict.get("intencion_esperada") or "",
        "bloqueado": False,
        "respuesta_estructurada": "",
        "fuentes_estructuradas": "",
        "respuesta_bruta": "",
        "fuentes_brutas": "",
        "trazador": trazador or "",
        "trazador_presente": "",
        "nota": pregunta_dict.get("nota", ""),
    }

    # ── Guardarraíl de inyección (categoría D) ────────────────────────────────
    if es_inyeccion_prompt(texto):
        fila["bloqueado"] = True
        fila["respuesta_estructurada"] = _MSG_INYECCION
        fila["respuesta_bruta"] = _MSG_INYECCION
        return fila

    # ── Respuestas de identidad / saludo (sin LLM ni retrieval) ──────────────
    resp_id = respuesta_identidad(texto)
    if resp_id:
        fila["respuesta_estructurada"] = resp_id
        fila["respuesta_bruta"] = resp_id
        return fila

    # ── Retrieval (una sola vez para ambos modos) ─────────────────────────────
    # hist_bruto se pasa para que _reescribir_con_contexto resuelva anáforas (C4).
    # Es el historial que siempre existe independientemente del plano.
    # Para preguntas independientes hist_bruto es [], sin efecto.
    intencion = detectar_intencion(texto)
    fila["intencion_detectada"] = intencion
    chunks, metas = buscar_contexto(coleccion, texto, historial=hist_bruto, intencion=intencion)

    if not chunks:
        fila["respuesta_estructurada"] = _MSG_SIN_COBERTURA
        fila["respuesta_bruta"] = _MSG_SIN_COBERTURA
        return fila

    # ── Generación: solo los modos que evalúa el plano ───────────────────────
    # plano "ambos" (C3): estructurado para comprobar veredicto + bruto para contenido.
    # cualquier otro plano: solo bruto (reduce tokens a la mitad).
    modos = ["estructurado", "bruto"] if plano == "ambos" else ["bruto"]

    for j, modo in enumerate(modos):
        hist = hist_est if modo == "estructurado" else hist_bruto
        resp = _recoger(
            generar_respuesta(
                chunks, metas, texto,
                historial=hist, intencion=intencion, modo_salida=modo,
            )
        )
        if modo == "estructurado":
            fila["respuesta_estructurada"] = resp
            fila["fuentes_estructuradas"] = resumen_fuentes(metas, neutro=False)
        else:
            fila["respuesta_bruta"] = resp
            fila["fuentes_brutas"] = resumen_fuentes(metas, neutro=True)
        if j < len(modos) - 1:
            time.sleep(PAUSA_ENTRE_MODOS)

    # ── Verificación de trazador (C2) ─────────────────────────────────────────
    if trazador and fila["respuesta_bruta"]:
        fila["trazador_presente"] = str(trazador.lower() in fila["respuesta_bruta"].lower())

    return fila


def ejecutar_experimento(
    salida_csv: str = SALIDA_CSV,
    filtro: list[str] | None = None,
) -> None:
    """
    filtro: lista de ids ("C2-1") o categorías ("C4", "D") a ejecutar.
    Si es None, se ejecutan todas las preguntas de la batería.
    """
    preguntas = PREGUNTAS
    if filtro:
        preguntas = [
            p for p in PREGUNTAS
            if p["id"] in filtro or p["categoria"] in filtro
        ]

    print("Cargando colección ChromaDB...")
    cliente = chromadb.PersistentClient(path=RUTA_BD)
    coleccion = obtener_coleccion(cliente)
    indexar_documentos(coleccion)
    print(f"Colección lista. {len(preguntas)} preguntas → {salida_csv}\n")

    # Historiales de hilos C4 (dos modos independientes)
    hilos_est: dict[str, list] = {}
    hilos_bruto: dict[str, list] = {}

    with open(salida_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=_CAMPOS_CSV)
        writer.writeheader()

        for i, pq in enumerate(preguntas, 1):
            texto = pq["texto"]
            hilo = pq.get("hilo")
            cat = pq["categoria"]
            print(f"[{i:02d}/{len(preguntas)}] [{cat}] {texto[:68]}{'...' if len(texto) > 68 else ''}")

            # Recuperar o iniciar historial del hilo
            hist_est = hilos_est.setdefault(hilo, []) if hilo else []
            hist_bruto = hilos_bruto.setdefault(hilo, []) if hilo else []

            for intento in range(3):
                try:
                    fila = _procesar(coleccion, pq, hist_est, hist_bruto)
                    break
                except RateLimitError:
                    print(f"         limite alcanzado; espero 60 s ({intento + 1}/3)")
                    time.sleep(60)
            else:
                print("         saltada tras 3 intentos\n")
                continue

            # Actualizar historial C4: solo el/los modos que se generaron
            if hilo and not fila["bloqueado"]:
                if fila["respuesta_estructurada"]:
                    hilos_est[hilo] += [
                        {"rol": "user", "contenido": texto},
                        {"rol": "assistant", "contenido": fila["respuesta_estructurada"]},
                    ]
                if fila["respuesta_bruta"]:
                    hilos_bruto[hilo] += [
                        {"rol": "user", "contenido": texto},
                        {"rol": "assistant", "contenido": fila["respuesta_bruta"]},
                    ]

            _log(fila)
            writer.writerow(fila)
            f.flush()

            if i < len(preguntas):
                time.sleep(PAUSA_ENTRE_PREGUNTAS)

    print(f"\nExperimento completado. CSV: {salida_csv}")


def _log(fila: dict) -> None:
    bloq = " BLOQUEADO" if fila["bloqueado"] else ""
    int_ = fila["intencion_detectada"] or "—"
    print(f"         intención={int_}{bloq}")
    if fila["trazador"] and fila["trazador_presente"]:
        ok = "OK" if fila["trazador_presente"] == "True" else "FALLO"
        print(f"         trazador '{fila['trazador']}': {ok}")
    est_chars = len(fila["respuesta_estructurada"])
    bru_chars = len(fila["respuesta_bruta"])
    if not fila["bloqueado"]:
        print(f"         estructurado={est_chars}c  bruto={bru_chars}c")
    print()


if __name__ == "__main__":
    destino, filtro = SALIDA_CSV, None
    for arg in sys.argv[1:]:
        if arg.endswith(".csv"):
            destino = arg
        else:
            filtro = arg.split(",")
    ejecutar_experimento(destino, filtro)
