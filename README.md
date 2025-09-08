# agent_skeletons_ollama

Skeletons de raciocínio (chain-of-thought **estrutural**) para agentes LLM usando **LangChain** e **LangGraph**, prontos para rodar **100% local** com **Ollama**. Retriever real: **Chroma** (FAISS removido).

## 🚀 Requisitos
- Python 3.10+
- [Ollama](https://ollama.com/) (`ollama serve`)
- Modelos (ex.: `qwen2.5:3b`, `llama3.1:8b`)
- Para Chroma: `pip install -e ".[rag]"`

## 🔧 Instalação
```bash
pip install -e ".[dev]"
pip install -e ".[rag]"  # para usar Chroma
```

## ⚙️ Variáveis de ambiente
```bash
export OLLAMA_MODEL="qwen2.5:3b"
export OLLAMA_EMBED_MODEL="nomic-embed-text"  # embeddings servidos pelo Ollama
export RETRIEVER_KIND="chroma"                 # ou "dummy"
export CHROMA_PERSIST_DIR=".chroma"            # opcional (persistência)
```

## ▶️ Exemplos
```bash
python examples/run_examples.py
```

## 🧪 Testes
```bash
pytest
```

## 🧰 Benchmarks
- `examples/bench_speed.py`: mede latência (TTFT, total) e tokens/s direto na API do Ollama.
- `examples/eval_ragas.py`: avalia qualidade no mini-RAG com **RAGAS**.

=== Comparação Latência vs Qualidade ===
Model           Total(s) TPS    Faith  Rel    CPrec  CRec  
qwen2.5:3b      4.95     49.9   0.91   0.86   1.00   1.00  
llama3.1:8b     13.30    23.7   0.71   0.69   1.00   1.00  

JSON bruto:
[
  {
    "model": "qwen2.5:3b",
    "ttft": 0.25728910839825403,
    "total": 4.950167633398086,
    "tps": 49.85260837945425,
    "tokens_avg": 246.2,
    "answer_relevancy": 0.8639558749924131,
    "faithfulness": 0.9068675383293529,
    "context_precision": 1.0,
    "context_recall": 1.0
  },
  {
    "model": "llama3.1:8b",
    "ttft": 0.3839979582000524,
    "total": 13.295807333002449,
    "tps": 23.738736983333116,
    "tokens_avg": 315.2,
    "answer_relevancy": 0.6931354415541303,
    "faithfulness": 0.7143784584711925,
    "context_precision": 1.0,
    "context_recall": 1.0
  }
]
Salvo em compare_results.csv


## 🧱 Skeletons incluídos
1. **Problema → Passos → Resposta** (Runnable + Graph)
2. **Contexto → Hipóteses → Evidências → Conclusão** (Runnable + Graph)
3. **Prós → Contras → Avaliação → Decisão** (Runnable + Graph)
4. **Entrada → Transformação → Saída (mini‑RAG)** com **Chroma** (Runnable + Graph)

> Observação: os skeletons organizam o raciocínio de forma **estrutural**, sem expor raciocínio privado passo a passo; foque no *formato de saída* e *estratégia*.
