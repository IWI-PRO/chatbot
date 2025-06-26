import subprocess

def query_ollama(prompt, model="llama3"):
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            text=True,
            capture_output=True,
            encoding="utf-8",         # ✅ Ensure input/output encoding is UTF-8
            errors="ignore"           # ✅ Ignore characters that can't be encoded
        )
        return result.stdout
    except Exception as e:
        return f"[Ollama Error] {e}"
