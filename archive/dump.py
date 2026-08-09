# create_dump.py
import subprocess
import pathlib
import os
from typing import Optional
# --- Configuration ---
PROJECT_ROOT = pathlib.Path('.') # Assumes script is run from project root
OUTPUT_FILENAME = "project_dump.txt"

# Directories/Files/Patterns to completely exclude from file search AND tree view
# Add any other specific directories or files you want to skip entirely.
# IMPORTANT: Ensure your virtual environment ('env', 'venv', '.venv', etc.) is listed here.
EXCLUDE_LIST = ['__pycache__', 'env', 'venv', '.venv', '.venv2', '.git', 'node_modules', '*.sqlite', '*.pyc','test','dump.py', OUTPUT_FILENAME]

# Optional: List specific *.py files or directories containing *.py files to include.
# If None, the script will include ALL .py files found (respecting EXCLUDE_LIST).
# Use this if you only want to share a subset. Paths are relative to PROJECT_ROOT.
# Example: INCLUDE_FILES_OR_DIRS = ['graph_builder.py', 'routing/', 'mocks/llm_mocks.py']
INCLUDE_FILES_OR_DIRS = None
# --- End Configuration ---
def get_tree_output(root: pathlib.Path, exclude_list: list[str]) -> str:
    """
    Runs the `tree` command with proper exclusion. This version works reliably.
    We combine all exclude patterns into a single -I "a|b|c" regex.
    """
    # Combine all patterns into a single regex, escaped properly
    # - Works for directory names ("env") and wildcard patterns ("*.sqlite")
    exclude_regex = "|".join(exclude_list)

    command = [
        "tree",
        "-L", "3",           # depth
        "-a",                # show hidden files
        "-I", exclude_regex  # exclude by regex
    ]

    try:
        print("Running:", " ".join(command))
        result = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            if "command not found" in result.stderr.lower():
                return "ERROR: `tree` command not found. Install it with brew/apt."
            print("Warning: Tree returned error:", result.stderr)

        return result.stdout.strip()

    except FileNotFoundError:
        return "ERROR: `tree` command not found. Install it with brew/apt."
    except Exception as e:
        return f"ERROR running tree: {e}"


def find_python_files(root: pathlib.Path, exclude_list: list[str], include_list: Optional[list[str]] = None) -> list[pathlib.Path]:
    """Finds Python files, respecting exclusions and optional inclusions."""
    found_files = []
    absolute_exclude_paths = {root.joinpath(p).resolve() for p in exclude_list if '*' not in p and '?' not in p and '[' not in p} # Resolve non-patterns

    files_to_process = []
    if include_list:
        # User specified exactly what to include
        for item_path_str in include_list:
            item_path = root.joinpath(item_path_str).resolve()
            if item_path.is_dir():
                files_to_process.extend(item_path.rglob('*.py'))
            elif item_path.is_file() and item_path.suffix == '.py':
                files_to_process.append(item_path)
    else:
        # Include all .py files found
        files_to_process = list(root.rglob('*.py'))

    # Filter the collected files
    for file_path in files_to_process:
        file_path_resolved = file_path.resolve()
        is_excluded = False
        # Check against resolved exclude directories/files
        for excluded_path in absolute_exclude_paths:
            if file_path_resolved == excluded_path or excluded_path in file_path_resolved.parents:
                is_excluded = True
                break
        if is_excluded:
            continue

        # Check against wildcard patterns in exclude_list (simple matching)
        relative_path_str = str(file_path.relative_to(root))
        for pattern in exclude_list:
             if '*' in pattern or '?' in pattern or '[' in pattern:
                  # Using simple filename matching for wildcards here
                  if file_path.match(pattern):
                      is_excluded = True
                      break
        if is_excluded:
             continue

        found_files.append(file_path)


    # Return sorted, unique list relative to root
    unique_relative_paths = sorted({p.relative_to(root) for p in found_files})
    return [root.joinpath(p) for p in unique_relative_paths]


def main():
    """Generates the project dump file."""
    print("Starting project dump generation...")
    output_parts = []

    # 1. Get tree output
    print("Generating file tree...")
    tree_output = get_tree_output(PROJECT_ROOT, EXCLUDE_LIST)
    output_parts.append("--- PROJECT STRUCTURE (`tree -L 3 -a -I ...`) ---")
    output_parts.append("```text")
    output_parts.append(tree_output.strip())
    output_parts.append("```")
    output_parts.append("\n" + "="*40 + "\n")

    # 2. Find Python files
    print("Finding Python files...")
    python_files = find_python_files(PROJECT_ROOT, EXCLUDE_LIST, INCLUDE_FILES_OR_DIRS)
    print(f"Found {len(python_files)} Python files to include:")
    for f in python_files: print(f"- {f.relative_to(PROJECT_ROOT)}")

    # 3. Read and format files
    print("\nReading and formatting files...")
    for file_path in python_files:
        relative_path_str = str(file_path.relative_to(PROJECT_ROOT))
        print(f"Processing: {relative_path_str}")
        try:
            # Prepend identifier line
            output_parts.append(f"\n--- START FILE: {relative_path_str} ---\n")
            output_parts.append("```python")
            # Read file content
            content = file_path.read_text(encoding='utf-8').strip()
            output_parts.append(content) # Add content
            # Append closing tags and end identifier
            output_parts.append("```")
            output_parts.append(f"\n--- END FILE: {relative_path_str} ---")
        except Exception as e:
            print(f"  ERROR reading file {relative_path_str}: {e}")
            output_parts.append(f"\n--- START FILE: {relative_path_str} ---\n")
            output_parts.append("```text")
            output_parts.append(f"!!! ERROR READING FILE: {e} !!!")
            output_parts.append("```")
            output_parts.append(f"\n--- END FILE: {relative_path_str} ---")

    # 4. Write to output file
    output_string = "\n".join(output_parts)
    print(f"\nWriting output to {OUTPUT_FILENAME} ({len(output_string)} characters)...")
    try:
        with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
            f.write(output_string)
        print(f"--- Successfully generated {OUTPUT_FILENAME} ---")
        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"IMPORTANT: Please OPEN and MANUALLY REVIEW '{OUTPUT_FILENAME}' NOW.")
        print("REMOVE or REDACT any sensitive information (API keys, passwords, secrets, PII)")
        print("BEFORE pasting its content into the chat.")
        print("Check the file size; if it's very large, it might exceed chat limits.")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    except Exception as e:
        print(f"--- ERROR writing output file: {e} ---")

if __name__ == "__main__":
    main()