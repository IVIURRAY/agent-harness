"""Filesystem tools for the file-organiser demo. All paths go through the Sandbox."""

from __future__ import annotations

import shutil

from ..sandbox import Sandbox, SandboxViolation
from .base import Tool, ToolError, truncate


def filesystem_tools(sandbox: Sandbox) -> list[Tool]:
    def _resolve(path: str):
        try:
            return sandbox.resolve(path)
        except SandboxViolation as exc:
            raise ToolError(str(exc)) from exc

    def list_files(path: str = ".") -> str:
        target = _resolve(path)
        if not target.is_dir():
            raise ToolError(f"Not a directory: {path}")
        lines = []
        for entry in sorted(target.rglob("*")):
            if entry.is_symlink():
                continue  # don't surface links; they could point anywhere
            rel = sandbox.rel(entry)
            if entry.is_dir():
                lines.append(f"{rel}/")
            else:
                lines.append(f"{rel}  ({entry.stat().st_size} bytes)")
        return "\n".join(lines) if lines else "(empty)"

    def read_file(path: str) -> str:
        target = _resolve(path)
        if not target.is_file():
            raise ToolError(f"Not a file: {path}")
        return truncate(target.read_text(encoding="utf-8", errors="replace"), 4_000)

    def create_directory(path: str) -> str:
        target = _resolve(path)
        if target.exists():
            raise ToolError(f"Already exists: {path}")
        target.mkdir(parents=True)
        return f"Created directory {sandbox.rel(target)}/"

    def delete_directory(path: str) -> str:
        target = _resolve(path)
        if target == sandbox.root:
            raise ToolError("Refusing to delete the sandbox root.")
        if not target.is_dir():
            raise ToolError(f"Not a directory: {path}")
        if any(target.iterdir()):
            raise ToolError(f"Directory not empty: {path} (only empty directories can be deleted)")
        target.rmdir()
        return f"Deleted empty directory {path}"

    def move_file(source: str, destination: str) -> str:
        src, dst = _resolve(source), _resolve(destination)
        if not src.is_file():
            raise ToolError(f"Source is not a file: {source}")
        if dst.is_dir():
            dst = dst / src.name
        if dst.exists():
            raise ToolError(f"Destination already exists: {sandbox.rel(dst)} (no overwrites)")
        if not dst.parent.is_dir():
            raise ToolError(f"Destination directory does not exist: {sandbox.rel(dst.parent)}")
        shutil.move(str(src), str(dst))
        return f"Moved {source} -> {sandbox.rel(dst)}"

    path_param = {"type": "string", "description": "Path relative to the workspace root."}
    return [
        Tool(
            name="list_files",
            description="Recursively list files and directories in the workspace (or a subdirectory).",
            parameters={"path": {**path_param, "description": "Directory to list; '.' for the root."}},
            fn=list_files,
        ),
        Tool(
            name="read_file",
            description="Read a text file's contents (truncated to 4000 characters).",
            parameters={"path": path_param},
            required=["path"],
            fn=read_file,
        ),
        Tool(
            name="create_directory",
            description="Create a subdirectory. Requires user approval; the user may decline.",
            parameters={"path": path_param},
            required=["path"],
            fn=create_directory,
            requires_approval=True,
            describe=lambda a: f"Create directory  {a.get('path')}/",
        ),
        Tool(
            name="delete_directory",
            description="Delete an EMPTY subdirectory. Requires user approval.",
            parameters={"path": path_param},
            required=["path"],
            fn=delete_directory,
            requires_approval=True,
            describe=lambda a: f"Delete empty directory  {a.get('path')}/",
        ),
        Tool(
            name="move_file",
            description=(
                "Move a file. If destination is an existing directory the file keeps its name. "
                "Never overwrites. Requires user approval; the user may decline."
            ),
            parameters={
                "source": {"type": "string", "description": "File to move, relative to the root."},
                "destination": {"type": "string", "description": "Target directory or file path."},
            },
            required=["source", "destination"],
            fn=move_file,
            requires_approval=True,
            describe=lambda a: f"Move  {a.get('source')}  ->  {a.get('destination')}",
        ),
    ]
