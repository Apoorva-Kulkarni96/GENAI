import httpx
import json

OLLAMA_URL = "http://localhost:11434"

async def stream_from_ollama(model, system_prompt, user_message):
    payload = {
        "model":  model,
        "system": system_prompt,
        "prompt": user_message,
        "stream": True,
    }

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", f"{OLLAMA_URL}/api/generate", json=payload) as response:

            if response.status_code != 200:
                error = await response.aread()
                yield {"error": f"Ollama error: {error.decode()}"}
                return

            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                    yield {
                        "token": chunk.get("response", ""),
                        "done":  chunk.get("done", False)
                    }
                    if chunk.get("done"):
                        break
                except json.JSONDecodeError:
                    continue