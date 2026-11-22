"""Command-line entry point. `video-to-context mcp` starts the MCP server."""

import sys


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args[:1] == ["mcp"]:
        from .mcp_server import main as run_server  # Avoids loading the video libraries.

        run_server()
        return 0

    from .run import main as run_pipeline

    return run_pipeline(args)
