import chromadb
from rich.panel import Panel

from .config import RUTA_BD, console
from .generator import preguntar
from .indexer import indexar_documentos, obtener_coleccion


def main() -> None:
    console.print(Panel("[bold cyan]Sistema RAG[/bold cyan]", expand=False))

    cliente = chromadb.PersistentClient(path=RUTA_BD)
    coleccion = obtener_coleccion(cliente)

    console.print("\n[bold]Verificando documentos...[/bold]")
    indexar_documentos(coleccion)
    console.print(f"\n[green]Base de conocimiento lista: {coleccion.count()} fragmentos[/green]\n")
    console.print("Escribe [bold]'salir'[/bold] para terminar.\n" + "-" * 60)

    while True:
        try:
            pregunta = input("\nPregunta: ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Saliendo...[/yellow]")
            break

        if pregunta.lower() == "salir":
            console.print("[yellow]Saliendo...[/yellow]")
            break

        if not pregunta:
            continue

        console.print("[dim]Buscando en documentación...[/dim]")
        preguntar(coleccion, pregunta)


if __name__ == "__main__":
    main()
