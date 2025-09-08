import json
import agent_skeletons.skeletons as sk

# DummyLLM to avoid real Ollama during tests
class DummyMsg:
    def __init__(self, content: str):
        self.content = content

class DummyLLM:
    def __init__(self, payload: str = "OK"):
        self.payload = payload
    def invoke(self, _msgs):
        return DummyMsg(self.payload)

def patch_llm(monkeypatch, payload="OK"):
    monkeypatch.setattr(sk, "get_llm", lambda: DummyLLM(payload))

def test_ppr_runnable(monkeypatch):
    patch_llm(monkeypatch, "- Passos:\n1. A\n2. B\n- Resposta: Done")
    chain = sk.build_problema_passos_resposta_chain()
    out = chain.invoke({"dominio":"eng dados","problema":"X","requisitos":"Y"})
    assert "Passos" in out and "Resposta" in out

def test_chec_json(monkeypatch):
    payload = json.dumps({
        "contexto":"Falha",
        "hipoteses":["dados","credenciais"],
        "evidencias":{"dados":["pro: X", "contra: Y"], "credenciais":["pro: Z"]},
        "conclusao":"dados"
    })
    patch_llm(monkeypatch, payload)
    chain = sk.build_contexto_hipoteses_chain()
    out = chain.invoke({"contexto":"Falhou"})
    data = json.loads(out)
    assert set(["contexto","hipoteses","evidencias","conclusao"]).issubset(data.keys())

def test_pcd_fields(monkeypatch):
    payload = json.dumps({
        "criterios":["latencia","custo"],
        "opcoes":["Weaviate","Pinecone"],
        "pros":{"Weaviate":["OSS"],"Pinecone":["SaaS"]},
        "contras":{"Weaviate":["operacao"],"Pinecone":["custo"]},
        "notas_por_criterio":{"latencia":{"Weaviate":7,"Pinecone":9},"custo":{"Weaviate":9,"Pinecone":6}},
        "decisao":"Weaviate na POC"
    })
    patch_llm(monkeypatch, payload)
    chain = sk.build_pros_contras_decisao_chain()
    out = chain.invoke({"cenario":"vector DB","opcoes":["Weaviate","Pinecone"],"criterios":["latencia","custo"]})
    data = json.loads(out)
    assert "decisao" in data and "notas_por_criterio" in data

def test_ets_graph(monkeypatch):
    patch_llm(monkeypatch, "Resposta: [doc1]")
    g = sk.build_ets_graph()
    out = g.invoke({"pergunta":"Qual a Selic em 2025?"})
    assert isinstance(out, dict) and "resposta" in out
