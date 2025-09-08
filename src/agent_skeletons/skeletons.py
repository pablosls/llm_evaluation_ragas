from __future__ import annotations
import os
from typing import TypedDict, List, Dict, Any
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# === Provider: OLLAMA ===
# Mantemos tudo local usando ChatOllama + OllamaEmbeddings

def get_llm():
    from langchain_ollama import ChatOllama
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    return ChatOllama(model=model, temperature=0.2)

def get_embeddings():
    from langchain_ollama import OllamaEmbeddings
    embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    return OllamaEmbeddings(model=embed_model)

# ======================================================================
# Retrievers reais: Chroma (com fallback Dummy) — FAISS removido
# ======================================================================
class DummyRetriever:
    def __init__(self, kb: Dict[str, str]):
        self.kb = kb
    def invoke(self, query: str):
        terms = set(query.lower().split())
        chunks = []
        for k, v in self.kb.items():
            score = sum(1 for t in terms if t in v.lower())
            if score:
                chunks.append({"id": k, "content": v, "score": score})
        return sorted(chunks, key=lambda x: -x["score"])[:5]

class ChromaRetriever:
    """Wrapper para Chroma que expõe .invoke(query) -> [{id, content}]"""
    def __init__(self, kb: Dict[str, str], persist_dir: str | None = None):
        self.embed = get_embeddings()
        from langchain_community.vectorstores import Chroma
        texts = list(kb.values())
        metadatas = [{"id": k} for k in kb.keys()]
        persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", ".chroma")
        self.store = Chroma.from_texts(
            texts=texts,
            embedding=self.embed,
            metadatas=metadatas,
            persist_directory=persist_dir,
        )
        try:
            self.store.persist()
        except Exception:
            pass

    def invoke(self, query: str, k: int = 5):
        docs = self.store.similarity_search(query, k=k)
        out = []
        for d in docs:
            doc_id = (d.metadata or {}).get("id", "doc")
            out.append({"id": doc_id, "content": d.page_content})
        return out

def make_retriever(kind: str, kb: Dict[str, str]):
    kind = (kind or "dummy").lower()
    if kind == "chroma":
        return ChromaRetriever(kb)
    return DummyRetriever(kb)

# ======================================================================
# 1) Problema → Passos → Resposta (Runnable)
# ======================================================================
PROBLEMA_PASSOS_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", "Você é um especialista em {dominio}. Organize o raciocínio em passos curtos e claros."),
    ("human", (
        "Problema: {problema}\n"
        "Requisitos: {requisitos}\n"
        "Formato de saída:\n"
        "- Passos: lista numerada de sub-etapas\n"
        "- Resposta: conclusão objetiva baseada nos passos\n"
    )),
])

def build_problema_passos_resposta_chain():
    llm = get_llm()
    return PROBLEMA_PASSOS_TEMPLATE | llm | StrOutputParser()

# ======================================================================
# 2) Contexto → Hipóteses → Evidências → Conclusão (Runnable)
# ======================================================================
class Diagnostico(BaseModel):
    contexto: str = Field(...)
    hipoteses: List[str]
    evidencias: Dict[str, List[str]]  # hipotese -> evidencias
    conclusao: str

C_H_E_C_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", "Você é um analista técnico. Produza JSON válido no schema solicitado."),
    ("human", (
        "Contexto: {contexto}\n"
        "Gere um diagnóstico no schema: {schema}\n"
        "Instruções:\n"
        "- Liste 2-4 hipóteses.\n"
        "- Em 'evidencias', para cada hipótese inclua bullets 'pro:' e 'contra:'.\n"
        "- Conclua indicando a hipótese mais provável e a ação recomendada.\n"
    )),
])

def build_contexto_hipoteses_chain():
    llm = get_llm()
    prompt = C_H_E_C_TEMPLATE.partial(schema=Diagnostico.model_json_schema())
    return prompt | llm | StrOutputParser()

# ======================================================================
# 3) Prós → Contras → Avaliação → Decisão (Runnable)
# ======================================================================
class Avaliacao(BaseModel):
    criterios: List[str]
    opcoes: List[str]
    pros: Dict[str, List[str]]
    contras: Dict[str, List[str]]
    notas_por_criterio: Dict[str, Dict[str, float]]
    decisao: str

PROS_CONTRAS_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", "Você é um consultor de arquitetura de IA. Compare opções, avalie critérios e decida."),
    ("human", (
        "Cenário: {cenario}\n"
        "Opções: {opcoes}\n"
        "Critérios (0..10): {criterios}\n"
        "Formato: JSON com chaves: criterios, opcoes, pros, contras, notas_por_criterio, decisao\n"
        "Observações:\n"
        "- Seja objetivo nos prós/contras.\n"
        "- Preencha 'notas_por_criterio' com uma nota para cada opção em cada critério.\n"
        "- 'decisao' deve explicar o trade-off.\n"
    )),
])

def build_pros_contras_decisao_chain():
    llm = get_llm()
    return PROS_CONTRAS_TEMPLATE | llm | StrOutputParser()

# ======================================================================
# 4) Entrada → Transformação → Saída (mini-RAG) (Runnable) com retriever Chroma
# ======================================================================
RAG_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", "Você é um agente RAG. Responda apenas com base nos contextos fornecidos."),
    ("human", (
        "Pergunta: {pergunta}\n"
        "Contextos:\n{contextos}\n"
        "Formato: resposta direta + cite ids dos contextos usados."
    )),
])

def build_entrada_transformacao_saida_chain():
    llm = get_llm()
    kb = {
        "doc1": "A taxa Selic é definida pelo Copom. Em 2025 variou entre X% e Y%.",
        "doc2": "O Banco Central publica a meta Selic após cada reunião.",
        "doc3": "Em set/2025, a meta Selic estava em Z% (ata do Copom).",
    }
    retriever_kind = os.getenv("RETRIEVER_KIND", "dummy")  # dummy | chroma
    retriever = make_retriever(retriever_kind, kb)

    def fetch_contexts(pergunta: str):
        docs = retriever.invoke(pergunta)
        ctx = "\n".join([f"[{d['id']}] {d['content']}" for d in docs])
        return {"pergunta": pergunta, "contextos": ctx}

    return (
        RunnableLambda(lambda x: x["pergunta"]) |
        RunnableLambda(fetch_contexts) |
        RAG_TEMPLATE |
        llm |
        StrOutputParser()
    )

# ======================================================================
# Versões LangGraph
# ======================================================================
from langgraph.graph import StateGraph, END

class PPRState(TypedDict):
    dominio: str
    problema: str
    requisitos: str
    resposta: str

def build_ppr_graph():
    llm = get_llm()
    def plan(state: PPRState):
        msgs = PROBLEMA_PASSOS_TEMPLATE.format_messages(
            dominio=state.get("dominio", "engenharia de dados"),
            problema=state["problema"],
            requisitos=state.get("requisitos", ""),
        )
        out = llm.invoke(msgs)
        return {"resposta": out.content}
    g = StateGraph(PPRState)
    g.add_node("planejar", plan)
    g.set_entry_point("planejar")
    g.add_edge("planejar", END)
    return g.compile()

class CHECState(TypedDict):
    contexto: str
    output: str

def build_chec_graph():
    llm = get_llm()
    def diagnose(state: CHECState):
        msgs = C_H_E_C_TEMPLATE.format_messages(
            contexto=state["contexto"],
            schema=Diagnostico.model_json_schema(),
        )
        out = llm.invoke(msgs)
        return {"output": out.content}
    g = StateGraph(CHECState)
    g.add_node("diagnosticar", diagnose)
    g.set_entry_point("diagnosticar")
    g.add_edge("diagnosticar", END)
    return g.compile()

class PCDState(TypedDict):
    cenario: str
    opcoes: List[str]
    criterios: List[str]
    output: str

def build_pcd_graph():
    llm = get_llm()
    def evaluate(state: PCDState):
        msgs = PROS_CONTRAS_TEMPLATE.format_messages(
            cenario=state["cenario"],
            opcoes=", ".join(state["opcoes"]),
            criterios=", ".join(state["criterios"]),
        )
        out = llm.invoke(msgs)
        return {"output": out.content}
    g = StateGraph(PCDState)
    g.add_node("avaliar", evaluate)
    g.set_entry_point("avaliar")
    g.add_edge("avaliar", END)
    return g.compile()

class ETSState(TypedDict):
    pergunta: str
    contextos: str
    resposta: str

def build_ets_graph():
    llm = get_llm()
    kb = {
        "doc1": "A taxa Selic é definida pelo Copom. Em 2025 variou entre X% e Y%.",
        "doc2": "O Banco Central publica a meta Selic após cada reunião.",
        "doc3": "Em set/2025, a meta Selic estava em Z% (ata do Copom).",
    }
    retriever_kind = os.getenv("RETRIEVER_KIND", "dummy")
    retriever = make_retriever(retriever_kind, kb)

    def retrieve(state: ETSState):
        docs = retriever.invoke(state["pergunta"])
        ctx = "\n".join([f"[{d['id']}] {d['content']}" for d in docs])
        return {"pergunta": state["pergunta"], "contextos": ctx}

    def answer(state: dict):
        msgs = RAG_TEMPLATE.format_messages(
            pergunta=state["pergunta"],
            contextos=state.get("contextos", ""),
        )
        out = llm.invoke(msgs)
        return {"resposta": out.content}

    g = StateGraph(ETSState)
    g.add_node("buscar", retrieve)
    g.add_node("responder", answer)
    g.set_entry_point("buscar")
    g.add_edge("buscar", "responder")
    g.add_edge("responder", END)
    return g.compile()
