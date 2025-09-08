import time, json, requests, statistics, os

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODELS = os.getenv("MODELS", "qwen2.5:3b,llama3.1:8b,mistral:7b").split(",")
PROMPT = os.getenv("PROMPT", "Explique em 4 bullets: como otimizar um job PySpark com shuffle pesado.")
N_RUNS = int(os.getenv("N_RUNS", "5"))
WARMUPS = int(os.getenv("WARMUPS", "1"))
OPTIONS = {
    "temperature": float(os.getenv("TEMP", "0.2")),
    "num_ctx": int(os.getenv("NUM_CTX", "4096")),
    "num_predict": int(os.getenv("NUM_PREDICT", "512")),
}

def chat_stream(model: str, prompt: str):
    url = f"{OLLAMA_URL}/api/chat"
    payload = {"model": model, "messages": [{"role":"user","content": prompt}], "stream": True, "options": OPTIONS}
    s = requests.post(url, json=payload, stream=True, timeout=None)
    s.raise_for_status()
    return s.iter_lines()

def time_one(model: str, prompt: str):
    t0 = time.perf_counter()
    ttft = None
    tokens = 0
    for line in chat_stream(model, prompt):
        if not line:
            continue
        data = json.loads(line.decode("utf-8"))
        if "message" in data and data["message"].get("content"):
            if ttft is None:
                ttft = time.perf_counter() - t0
            tokens += len(data["message"]["content"].split())
        if data.get("done"):
            break
    total = time.perf_counter() - t0
    tps = tokens / total if total > 0 else 0.0
    return {"ttft": ttft or total, "total": total, "tokens": tokens, "tps": tps}

def run_model(model: str):
    for _ in range(WARMUPS):
        try: time_one(model, "warmup, ignore.")
        except Exception: pass
    results = [time_one(model, PROMPT) for _ in range(N_RUNS)]
    def agg(key):
        xs = [r[key] for r in results]
        return {
            "avg": statistics.mean(xs),
            "p50": statistics.median(xs),
            "p90": statistics.quantiles(xs, n=10)[8],
            "std": statistics.pstdev(xs),
        }
    return {
        "model": model,
        "ttft": agg("ttft"),
        "total": agg("total"),
        "tps": agg("tps"),
        "tokens_avg": statistics.mean([r["tokens"] for r in results]),
    }

def main():
    table = []
    for m in MODELS:
        try:
            table.append(run_model(m))
        except Exception as e:
            table.append({"model": m, "error": str(e)})
    ok = [x for x in table if "error" not in x]
    ok.sort(key=lambda r: r["total"]["avg"])
    print("\n=== Ranking por latência média (total.avg) ===")
    for r in ok:
        print(f"{r['model']:<15} total.avg={r['total']['avg']:.3f}s  ttft.avg={r['ttft']['avg']:.3f}s  tps.avg={r['tps']['avg']:.1f} tok/s")
    print("\nJSON bruto:")
    print(json.dumps(table, indent=2))

if __name__ == "__main__":
    main()
