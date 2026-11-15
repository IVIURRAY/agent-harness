"""Filesystem jail. Every path the model supplies goes through Sandbox.resolve().

Enforcement lives in code, not in the prompt: telling the model "only touch the demo
directory" is a hint; this is the guarantee.
"""

from __future__ import annotations

import os
from pathlib import Path


class SandboxViolation(Exception):
    """Raised when a path would escape the sandbox root."""


class Sandbox:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError(f"Sandbox root is not a directory: {self.root}")

    def resolve(self, user_path: str) -> Path:
        if user_path is None or "\x00" in user_path:
            raise SandboxViolation("Invalid path")
        candidate = Path(user_path)
        # Treat absolute paths as relative to the root ("/notes" -> "<root>/notes").
        if candidate.is_absolute():
            candidate = Path(*candidate.parts[1:]) if len(candidate.parts) > 1 else Path(".")
        # resolve() follows symlinks, so a link pointing outside is caught too.
        resolved = (self.root / candidate).resolve(strict=False)
        if os.path.commonpath([self.root, resolved]) != str(self.root):
            raise SandboxViolation(f"Path escapes the sandbox: {user_path!r}")
        return resolved

    def rel(self, path: Path) -> str:
        """Path as shown to the model and the user: relative to the root."""
        relative = path.resolve().relative_to(self.root)
        return str(relative) if str(relative) != "." else "."
