"""Repository ingestion and code-aware parsing module using tree-sitter.

Supports:
- Cloning GitHub repositories via GitPython
- Parsing source code files using tree-sitter for Java, Python, JS, TS, C, C++
- Preserving file path, symbol name, and 1-based start/end line numbers
- Structured section chunking for Markdown, JSON, YAML, and XML
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional

import git
from tree_sitter import Language, Node, Parser
import tree_sitter_c
import tree_sitter_cpp
import tree_sitter_java
import tree_sitter_javascript
import tree_sitter_python
import tree_sitter_typescript

# Initialize parsers for supported compiled/scripted languages
LANGUAGE_PARSERS: dict[str, Parser] = {
    "python": Parser(Language(tree_sitter_python.language())),
    "java": Parser(Language(tree_sitter_java.language())),
    "javascript": Parser(Language(tree_sitter_javascript.language())),
    "typescript": Parser(Language(tree_sitter_typescript.language_typescript())),
    "c": Parser(Language(tree_sitter_c.language())),
    "cpp": Parser(Language(tree_sitter_cpp.language())),
}

# Mapping of file extensions to language identifier
EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".java": "java",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".md": "markdown",
    ".markdown": "markdown",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".xml": "xml",
}

# Directories and files to ignore during ingestion
IGNORED_DIRS = {
    ".git",
    ".github",
    ".vscode",
    ".idea",
    "__pycache__",
    "node_modules",
    "venv",
    ".venv",
    "env",
    ".env",
    "target",
    "build",
    "dist",
    ".next",
    "out",
    "bin",
    "obj",
}

IGNORED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".ico",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".class",
    ".pyc",
    ".pyd",
    ".lock",
}


def clone_or_open_repository(repo_url_or_path: str, destination_base_dir: str = "data/repositories") -> tuple[str, str, str]:
    """Clone a remote GitHub repo or open an existing local repo directory.

    Args:
        repo_url_or_path: Remote URL (https://github.com/...) or local directory path.
        destination_base_dir: Base directory where cloned repositories are stored.

    Returns:
        tuple of (repo_name, local_repo_path, head_commit_hash)
    """
    os.makedirs(destination_base_dir, exist_ok=True)

    # Check if input is a local existing directory
    if os.path.isdir(repo_url_or_path):
        local_path = os.path.abspath(repo_url_or_path)
        repo_name = os.path.basename(local_path)
        try:
            repo = git.Repo(local_path)
            head_commit = repo.head.commit.hexsha
        except Exception:
            head_commit = "local_uncommitted"
        return repo_name, local_path, head_commit

    # Parse repo name from URL (e.g., https://github.com/owner/my-repo.git -> my-repo)
    clean_url = repo_url_or_path.rstrip("/")
    if clean_url.endswith(".git"):
        clean_url = clean_url[:-4]
    repo_name = clean_url.split("/")[-1]

    target_dir = os.path.abspath(os.path.join(destination_base_dir, repo_name))

    if os.path.exists(target_dir):
        # Open existing repo
        repo = git.Repo(target_dir)
        try:
            repo.remotes.origin.pull()
        except Exception:
            pass  # Offline or local changes, proceed with existing files
    else:
        # Clone fresh repo
        repo = git.Repo.clone_from(repo_url_or_path, target_dir)

    head_commit = repo.head.commit.hexsha
    return repo_name, target_dir, head_commit


def get_node_name(node: Node, source_bytes: bytes) -> Optional[str]:
    """Extract symbol name from AST node."""
    name_node = node.child_by_field_name("name")
    if name_node:
        return source_bytes[name_node.start_byte : name_node.end_byte].decode("utf-8", errors="ignore")

    # In C/C++, declarator holds the function or variable name
    decl_node = node.child_by_field_name("declarator")
    if decl_node:
        # Recursively find the identifier in declarators (e.g., function_declarator)
        curr = decl_node
        while curr and curr.child_by_field_name("declarator"):
            curr = curr.child_by_field_name("declarator")
        ident = curr.child_by_field_name("name") if curr else None
        if ident:
            return source_bytes[ident.start_byte : ident.end_byte].decode("utf-8", errors="ignore")
        # Fallback to first identifier token in declarator
        for child in curr.children if curr else []:
            if child.type == "identifier":
                return source_bytes[child.start_byte : child.end_byte].decode("utf-8", errors="ignore")
        return source_bytes[decl_node.start_byte : decl_node.end_byte].decode("utf-8", errors="ignore")

    return None


def extract_tree_sitter_chunks(
    source_code: str,
    language: str,
    repo_name: str,
    file_path: str,
) -> list[dict[str, Any]]:
    """Parse source code with tree-sitter and extract class/method/function chunks.

    Preserves:
    - file_path
    - symbol_name (e.g., 'UserService', 'authenticate', 'UserService.authenticate')
    - line numbers (1-based start_line, end_line)
    """
    parser = LANGUAGE_PARSERS.get(language)
    if not parser:
        return []

    source_bytes = source_code.encode("utf-8")
    tree = parser.parse(source_bytes)
    chunks: list[dict[str, Any]] = []

    # Target node types for each language
    class_types = {"class_definition", "class_declaration", "class_specifier", "interface_declaration", "struct_specifier", "record_declaration", "enum_declaration"}
    func_types = {"function_definition", "function_declaration", "method_declaration", "method_definition", "constructor_declaration"}

    lines = source_code.splitlines()
    total_lines = len(lines)

    def traverse(node: Node, parent_class: Optional[str] = None):
        node_type = node.type

        if node_type in class_types:
            cls_name = get_node_name(node, source_bytes) or "AnonymousClass"
            start_line = node.start_point.row + 1
            end_line = node.end_point.row + 1
            node_text = source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")

            # Add class-level chunk
            chunk_id = f"{repo_name}:{file_path}:{start_line}:{end_line}"
            chunks.append({
                "id": chunk_id,
                "repo_name": repo_name,
                "file_path": file_path,
                "language": language,
                "chunk_type": "class",
                "symbol_name": cls_name,
                "start_line": start_line,
                "end_line": end_line,
                "content": f"// File: {file_path} (Lines {start_line}-{end_line})\n// Class: {cls_name}\n{node_text}",
            })

            # Traverse inside class body for methods
            for child in node.children:
                traverse(child, parent_class=cls_name)
            return

        elif node_type in func_types:
            func_name = get_node_name(node, source_bytes) or "anonymous_function"
            full_symbol = f"{parent_class}.{func_name}" if parent_class else func_name
            start_line = node.start_point.row + 1
            end_line = node.end_point.row + 1
            node_text = source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")

            chunk_id = f"{repo_name}:{file_path}:{start_line}:{end_line}"
            chunks.append({
                "id": chunk_id,
                "repo_name": repo_name,
                "file_path": file_path,
                "language": language,
                "chunk_type": "method" if parent_class else "function",
                "symbol_name": full_symbol,
                "start_line": start_line,
                "end_line": end_line,
                "content": f"// File: {file_path} (Lines {start_line}-{end_line})\n// Symbol: {full_symbol}\n{node_text}",
            })
            return

        for child in node.children:
            traverse(child, parent_class=parent_class)

    traverse(tree.root_node)

    # Fallback if no classes or functions were found (or for short scripts)
    if not chunks and total_lines > 0:
        chunk_id = f"{repo_name}:{file_path}:1:{total_lines}"
        chunks.append({
            "id": chunk_id,
            "repo_name": repo_name,
            "file_path": file_path,
            "language": language,
            "chunk_type": "file",
            "symbol_name": Path(file_path).name,
            "start_line": 1,
            "end_line": total_lines,
            "content": f"// File: {file_path} (Lines 1-{total_lines})\n{source_code}",
        })

    return chunks


def extract_markdown_chunks(
    content: str,
    repo_name: str,
    file_path: str,
) -> list[dict[str, Any]]:
    """Chunk Markdown files by section headers (#, ##, ###)."""
    lines = content.splitlines()
    if not lines:
        return []

    sections = []
    current_header = Path(file_path).name
    current_lines: list[str] = []
    start_line = 1

    header_regex = re.compile(r"^(#{1,4})\s+(.+)$")

    for idx, line in enumerate(lines, 1):
        match = header_regex.match(line)
        if match and current_lines:
            end_line = idx - 1
            sections.append({
                "id": f"{repo_name}:{file_path}:{start_line}:{end_line}",
                "repo_name": repo_name,
                "file_path": file_path,
                "language": "markdown",
                "chunk_type": "section",
                "symbol_name": current_header,
                "start_line": start_line,
                "end_line": end_line,
                "content": f"<!-- File: {file_path} (Lines {start_line}-{end_line}) Section: {current_header} -->\n"
                + "\n".join(current_lines),
            })
            current_header = match.group(2).strip()
            current_lines = [line]
            start_line = idx
        else:
            if match:
                current_header = match.group(2).strip()
            current_lines.append(line)

    if current_lines:
        end_line = len(lines)
        sections.append({
            "id": f"{repo_name}:{file_path}:{start_line}:{end_line}",
            "repo_name": repo_name,
            "file_path": file_path,
            "language": "markdown",
            "chunk_type": "section",
            "symbol_name": current_header,
            "start_line": start_line,
            "end_line": end_line,
            "content": f"<!-- File: {file_path} (Lines {start_line}-{end_line}) Section: {current_header} -->\n"
            + "\n".join(current_lines),
        })

    return sections


def extract_structured_data_chunks(
    content: str,
    language: str,
    repo_name: str,
    file_path: str,
    max_lines_per_chunk: int = 50,
    overlap_lines: int = 10,
) -> list[dict[str, Any]]:
    """Chunk JSON, YAML, and XML files into structured logical slices."""
    lines = content.splitlines()
    if not lines:
        return []

    total_lines = len(lines)
    if total_lines <= max_lines_per_chunk:
        return [{
            "id": f"{repo_name}:{file_path}:1:{total_lines}",
            "repo_name": repo_name,
            "file_path": file_path,
            "language": language,
            "chunk_type": "data_file",
            "symbol_name": Path(file_path).name,
            "start_line": 1,
            "end_line": total_lines,
            "content": f"// File: {file_path} (Lines 1-{total_lines})\n{content}",
        }]

    chunks = []
    step = max(1, max_lines_per_chunk - overlap_lines)
    for start_idx in range(0, total_lines, step):
        end_idx = min(start_idx + max_lines_per_chunk, total_lines)
        chunk_lines = lines[start_idx:end_idx]
        start_line = start_idx + 1
        end_line = end_idx

        chunks.append({
            "id": f"{repo_name}:{file_path}:{start_line}:{end_line}",
            "repo_name": repo_name,
            "file_path": file_path,
            "language": language,
            "chunk_type": "data_block",
            "symbol_name": f"{Path(file_path).name}:{start_line}-{end_line}",
            "start_line": start_line,
            "end_line": end_line,
            "content": f"// File: {file_path} (Lines {start_line}-{end_line})\n" + "\n".join(chunk_lines),
        })

        if end_idx >= total_lines:
            break

    return chunks


def ingest_file(local_file_path: str, repo_root: str, repo_name: str) -> list[dict[str, Any]]:
    """Ingest a single repository file into chunks based on language."""
    rel_path = os.path.relpath(local_file_path, repo_root).replace("\\", "/")
    ext = Path(local_file_path).suffix.lower()
    language = EXTENSION_MAP.get(ext)

    if not language:
        return []

    try:
        with open(local_file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception:
        return []

    if not content.strip():
        return []

    if language in LANGUAGE_PARSERS:
        return extract_tree_sitter_chunks(content, language, repo_name, rel_path)
    elif language == "markdown":
        return extract_markdown_chunks(content, repo_name, rel_path)
    elif language in {"json", "yaml", "xml"}:
        return extract_structured_data_chunks(content, language, repo_name, rel_path)
    else:
        return []


def ingest_repository(repo_root: str, repo_name: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Traverse repository, filtering out unwanted folders, and extract all chunks.

    Returns:
        tuple of (all_chunks, indexed_file_paths)
    """
    all_chunks: list[dict[str, Any]] = []
    indexed_files: list[str] = []

    for root, dirs, files in os.walk(repo_root):
        # Exclude ignored directories in-place so os.walk does not traverse them
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and not d.startswith(".")]

        for file_name in files:
            ext = Path(file_name).suffix.lower()
            if ext in IGNORED_EXTENSIONS or ext not in EXTENSION_MAP:
                continue

            full_path = os.path.join(root, file_name)
            rel_path = os.path.relpath(full_path, repo_root).replace("\\", "/")

            file_chunks = ingest_file(full_path, repo_root, repo_name)
            if file_chunks:
                all_chunks.extend(file_chunks)
                indexed_files.append(rel_path)

    return all_chunks, indexed_files

