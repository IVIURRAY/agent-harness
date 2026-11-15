import os

import pytest

from harness.sandbox import Sandbox, SandboxViolation
from harness.tools import ToolError, filesystem_tools


@pytest.fixture
def box(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "a.txt").write_text("hello")
    (tmp_path / "secret.txt").write_text("nope")
    return Sandbox(root)


@pytest.mark.parametrize("path", ["../secret.txt", "sub/../../secret.txt", "../../etc/passwd"])
def test_traversal_blocked(box, path):
    with pytest.raises(SandboxViolation):
        box.resolve(path)


def test_absolute_paths_are_rooted(box):
    assert box.resolve("/a.txt") == box.root / "a.txt"


def test_symlink_escape_blocked(box, tmp_path):
    os.symlink(tmp_path / "secret.txt", box.root / "link.txt")
    with pytest.raises(SandboxViolation):
        box.resolve("link.txt")


def test_tools_surface_violations_as_tool_errors(box):
    tools = {t.name: t for t in filesystem_tools(box)}
    with pytest.raises(ToolError):
        tools["read_file"].fn(path="../secret.txt")
    with pytest.raises(ToolError):
        tools["move_file"].fn(source="a.txt", destination="../stolen.txt")
    with pytest.raises(ToolError):
        tools["delete_directory"].fn(path=".")


def test_move_never_overwrites(box):
    tools = {t.name: t for t in filesystem_tools(box)}
    (box.root / "b.txt").write_text("b")
    with pytest.raises(ToolError):
        tools["move_file"].fn(source="a.txt", destination="b.txt")
