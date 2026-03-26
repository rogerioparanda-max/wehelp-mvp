import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="WeHelp Retention MVP", layout="wide")

# =========================
# Helpers
# =========================

def md_join(lines):
    return "\n".join([str(x) for x in lines if str(x).strip()])

def find_comment_examples(df, touchpoint=None, nps_class=None, limit=2):
    if df is None or df.empty:
        return []
    d = df.copy()
    if touchpoint:
        d = d[d["touchpoint"] == touchpoint]
    if nps_class:
        d = d[d["class"] == nps_class]
    return d["comment"].dropna().astype(str).head(limit).tolist()

# =========================
# CORE AI
# =========================

def answer_question(q, insights):

    q = q.lower()

    selected_unit = insights["selected_unit"]
    overall_nps = insights["overall_nps"]
    zone = insights["zone"]
    benchmark_gap = insights["benchmark_gap"]
    network_nps = insights["network_nps"]
    unit_vs_network_gap = insights["unit_vs_network_gap"]
    priorities = insights["priorities"]
    comment_evidence = insights["comments"]

    def executive_intro():
        return f"""**Leitura executiva da unidade**

Unidade: **{selected_unit}**
NPS: **{overall_nps}** ({zone})
Gap vs benchmark: **{benchmark_gap}**
Gap vs rede: **{unit_vs_network_gap}**
"""

    # =========================
    # CONSULTORIA REAL
    # =========================

    if any(x in q for x in ["melhorar", "nps"]):

        if priorities.empty:
            return "Sem dados suficientes para análise."

        main_issue = priorities.iloc[0]["touchpoint"]

        examples = find_comment_examples(
            comment_evidence,
            touchpoint=main_issue,
            nps_class="Detrator",
            limit=2
        )

        evidence_text = (
            "\n".join([f'- "{e}"' for e in examples])
            if examples
            else "Sem comentários suficientes — mas o padrão já é claro."
        )

        return f"""{executive_intro()}

**Diagnóstico**

Você não tem vários problemas.

Você tem um principal:
→ **{main_issue}**

Esse ponto está puxando seu NPS para baixo.

---

**Por que isso acontece**

Não é um problema isolado.

É **falta de padrão de execução**.

Clientes percebem inconsistência.

---

**O que você deve fazer HOJE**

Vá para **{main_issue}** e observe:

- O que deveria acontecer vs o que acontece
- Onde o padrão quebra
- Quem executa bem vs mal

👉 Sem isso, você está gerenciando no escuro

---

**O que ajustar com o time**

Defina:

1. O padrão correto
2. O que é erro
3. Como o cliente percebe

👉 Sem padrão, cada funcionário faz de um jeito

---

**Voz do cliente**

{evidence_text}

---

**Leitura estratégica**

Você está **{abs(unit_vs_network_gap)} pontos da rede**

👉 Isso significa:

- operação inconsistente
- experiência irregular
- risco competitivo

---

**Resumo executivo**

Pare de tentar melhorar tudo.

Se você corrigir **{main_issue} com disciplina**, seu NPS sobe.

Se não, você continua rodando em falso.
"""

    return "Pergunta não reconhecida."

# =========================
# MOCK DATA (substituir depois)
# =========================

data = pd.DataFrame({
    "touchpoint": ["ATENDIMENTO NA RECEPÇÃO", "MUSCULAÇÃO", "TREINOS"],
    "nps_touchpoint": [40, 50, 45]
})

comments = pd.DataFrame({
    "touchpoint": ["ATENDIMENTO NA RECEPÇÃO", "ATENDIMENTO NA RECEPÇÃO"],
    "class": ["Detrator", "Detrator"],
    "comment": [
        "Demora no atendimento",
        "Funcionários não ajudam"
    ]
})

insights = {
    "selected_unit": "SELFIT - ARENA",
    "overall_nps": 66.7,
    "zone": "Zona de aperfeiçoamento",
    "benchmark_gap": 16.7,
    "network_nps": 62.6,
    "unit_vs_network_gap": 4.1,
    "priorities": data,
    "comments": comments
}

# =========================
# UI
# =========================

st.title("WeHelp Retention MVP")

q = st.text_input("Pergunta")

if st.button("Responder"):
    st.markdown(answer_question(q, insights))
