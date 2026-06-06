"""Script de diagnóstico para detectar_intencion."""
from src.retriever import detectar_intencion, _RE_NORMA, _RE_INTERNA, _RE_COMPARAR

casos = [
    ("A1", "¿Qué palabra deben contener las contraseñas de usuarios estándar?"),
    ("A2", "según mi normativa interna, qué palabra debe contener la contraseña de un usuario estándar?"),
    ("A3", "¿Qué estándar de la ISO 27002 trata la información de autenticación?"),
    ("Control1", "¿Cuál es el estándar de contraseñas de nuestra política?"),
    ("Control2", "¿qué dice el control 5.15 de la ISO 27002?"),
    ("Control3", "¿cuáles son los requisitos de contraseñas en mi empresa?"),
]

print(f"{'ID':<10} {'NORMA':<8} {'INTERNA':<10} {'COMPARAR':<10} {'INTENCION':<15} PREGUNTA")
print("-" * 120)
for id_caso, pregunta in casos:
    m_norma = bool(_RE_NORMA.search(pregunta))
    m_interna = bool(_RE_INTERNA.search(pregunta))
    m_comparar = bool(_RE_COMPARAR.search(pregunta))
    intencion = detectar_intencion(pregunta)
    print(f"{id_caso:<10} {str(m_norma):<8} {str(m_interna):<10} {str(m_comparar):<10} {intencion:<15} {pregunta[:60]}")

print()
print("Detalle de qué dispara _RE_NORMA en A1:")
m = _RE_NORMA.search("¿Qué palabra deben contener las contraseñas de usuarios estándar?")
if m:
    print(f"  Match: '{m.group()}' en posicion {m.start()}-{m.end()}")
else:
    print("  No hay match (esto seria lo esperado tras el fix)")