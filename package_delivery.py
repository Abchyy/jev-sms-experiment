"""Build a shareable archive without virtualenv, caches, secrets, or VCS files."""
import hashlib
import json
import re
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parent
excluded = {".venv", ".uv-cache", ".mpl-cache", "__pycache__", ".pytest_cache", ".git"}
files = []
for path in sorted(root.rglob("*")):
    rel = path.relative_to(root)
    if not path.is_file() or any(p in excluded for p in rel.parts):
        continue
    if path.name.startswith(".env") and path.name != ".env.example":
        continue
    if path.name == "SHA256SUMS.txt":
        continue
    data = path.read_bytes()
    if re.search(rb"sk-or-v1-[a-fA-F0-9]{64}", data):
        raise RuntimeError(f"Possible secret in {rel}; refusing archive")
    files.append((path, rel, hashlib.sha256(data).hexdigest()))
sums = root / "SHA256SUMS.txt"
sums.write_text("".join(f"{sha}  {rel}\n" for _, rel, sha in files))
archive = root.parent / "jev-sms-experiment-delivery.zip"
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
    for path, rel, _ in files:
        z.write(path, Path(root.name) / rel)
    z.write(sums, Path(root.name) / sums.name)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert not any(Path(name).name == ".env" for name in z.namelist())
print(json.dumps({"archive": str(archive), "files": len(files) + 1, "bytes": archive.stat().st_size,
                  "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "secrets_included": False}, indent=2))
