"""
_fix_unicode.py
Replace Unicode symbols that break Windows cp1252 terminal encoding.
Run once: python _fix_unicode.py
"""
import os
import pathlib

REPLACEMENTS = [
    ("\u2713", "[OK]"),       # ✓
    ("\u2717", "[FAIL]"),     # ✗
    ("\u26a0", "[WARN]"),     # ⚠
    ("\u2192", "->"),         # →
    ("\u2265", ">="),         # ≥
    ("\u2014", "-"),          # —  em dash
    ("\u2013", "-"),          # –  en dash
    ("\u00b1", "+-"),         # ±
    ("\u00d7", "x"),          # ×
    ("\u2019", "'"),          # '
    ("\u201c", '"'),          # "
    ("\u201d", '"'),          # "
]

root = pathlib.Path(__file__).parent

for py_file in root.glob("*.py"):
    if py_file.name.startswith("_"):
        continue  # skip this script itself

    text = py_file.read_text(encoding="utf-8")
    original = text

    for old, new in REPLACEMENTS:
        text = text.replace(old, new)

    if text != original:
        py_file.write_text(text, encoding="utf-8")
        print(f"Fixed: {py_file.name}")
    else:
        print(f"Clean: {py_file.name}")

print("Done.")
