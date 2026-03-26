import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="WeHelp Retention MVP", layout="wide")


# -----------------------------
# Helpers
# -----------------------------

def normalize_text(value: str) -> str:
    if value is None:
        return ""
    return str(value).strip()


def to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def classify_nps(score: float) -> str:
    if pd.isna(score):
        return "Sem nota"
    if score >= 9:
        return "Promotor"
    if score >= 7:
        return "Passivo"
    return "Detrator"


def nps_score(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if len(vals) == 0:
        return np.nan
    promoters = (vals >= 9).mean() * 100
    detractors = (vals <= 6).mean() * 100
    return promoters - detractors


def nps_zone(nps: float) -> str:
    if pd.isna(nps):
        return "Sem dados"
    if nps < 0:
        return "Crise"
    if nps < 50:
        return "Zona de aprendizado"
    if nps < 75:
        return "Zona de aperfeiçoamento"
    return "Zona de excelência"


def age_band(age: float) -> str:
    if pd.isna(age):
        return "Sem idade"
    bands = [(18, 25), (26, 30), (31, 35), (36, 40), (41, 45), (46, 50), (51, 55), (56, 60)]
    for start, end in bands:
        if start <= age <= end:
            return f"{start}-{end}"
    if age < 18:
        return "<18"
    return "61+"


def tenure_band(months: float) -> str:
    if pd.isna(months):
        return "Sem contrato"
    if months < 1:
        return "<1 mês"
    if months <= 3:
        return "1-3 meses"
    if months <= 6:
        return "4-6 meses"
    if months <= 12:
        return "7-12 meses"
    if months <= 24:
        return "13-24 meses"
    return "25+ meses"


def day_period(ts: pd.Timestamp) -> str:
    if pd.isna(ts):
        return "Sem horário"
    hour = ts.hour
    if 5 <= hour < 12:
        return "Manhã"
    if 12 <= hour < 18:
        return "Tarde"
    return "Noite"


def safe_bool_text(v) -> str:
    text = normalize_text(v).lower()
    if text in {"yes", "sim", "true", "1"}:
        return "Sim"
    if text in {"no", "não", "nao", "false", "0"}:
        return "Não"
    return normalize_text(v) if normalize_text(v) else "Não informado"


def extract_type_and_phrases(text: str) -> List[Tuple[str, str]]:
    if not text or pd.isna(text):
        return []
    chunks = []
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^(COMPLIMENT|COMPLAINT|SUGGESTION)\s*:\s*(.*)$", line, flags=re.I)
        if match:
            chunks.append((match.group(1).upper(), match.group(2).strip()))
        else:
            chunks.append(("UNKNOWN", line))
    return chunks


def odds_ratio_topbox(df: pd.DataFrame, col: str) -> float:
    temp = df[["nps_class", col]].copy()
    temp[col] = pd.to_numeric(temp[col], errors="coerce")
    temp = temp[temp["nps_class"].isin(["Promotor", "Detrator"]) & temp[col].notna()]
    if temp.empty:
        return np.nan
    temp["top_box"] = temp[col] >= 9

    promo = temp[temp["nps_class"] == "Promotor"]
    detra = temp[temp["nps_class"] == "Detrator"]
    if promo.empty or detra.empty:
        return np.nan

    a = promo["top_box"].sum()
    b = (~promo["top_box"]).sum()
    c = detra["top_box"].sum()
    d = (~detra["top_box"]).sum()

    # Haldane-Anscombe correction for zeros
    a, b, c, d = [x + 0.5 for x in (a, b, c, d)]
    return (a / b) / (c / d)


@dataclass
class Schema:
    nps: str
    nps_comment: Optional[str]
    unit: str
    response_unit: Optional[str]
    birth_date: Optional[str]
    gender: Optional[str]
    signup_date: Optional[str]
    frequency_date: Optional[str]
    had_problem: Optional[str]
    solved_problem: Optional[str]
    problem_eval: Optional[str]
    problem_comment: Optional[str]
    touchpoint_eval_cols: List[str]
    touchpoint_comment_cols: List[str]
    tag_cols: List[str]


DEFAULT_BENCHMARK = 50.0


# -----------------------------
# Schema detection
# -----------------------------

def detect_schema(df: pd.DataFrame) -> Schema:
    cols = list(df.columns)

    # Touchpoints
    touch_eval = [c for c in cols if str(c).endswith(" Evaluation")]
    touch_comment = [c for c in cols if str(c).endswith(" Comment")]

    # Tags now start from column AR (user update)
    # We'll take everything AFTER the last known column block as tags
    # Safer: exclude known columns and treat the rest as tags
    known_non_tag = {
        "Internal Code", "Name", "Phone", "Email", "Document",
        "Person Created At", "Date Of Birth", "Gender",
        "Person Company Unit Name", "Response Company Unit Name",
        "Country Name", "State Name",
        "Evaluation", "Nps Comment", "Authorization", "Nps Status",
        "Had Problem", "Solved Problem", "Evaluation Problem", "Comment Problem",
        "Frequency Date", "Messages",
        "Data Fim Plano", "Data Inicio Plano", "Nome do Plano",
        "Professor", "Tipo do Plano", "Valor do Plano",
        "Para finalizar, como você avalia a temperatura na academia durante sua visita?"
    }
    known_non_tag.update(touch_eval)
    known_non_tag.update(touch_comment)

    tag_cols = [c for c in cols if c not in known_non_tag]

    return Schema(
        nps="Evaluation",
        nps_comment="Nps Comment" if "Nps Comment" in cols else None,
        unit="Person Company Unit Name" if "Person Company Unit Name" in cols else "Response Company Unit Name",
        response_unit="Response Company Unit Name" if "Response Company Unit Name" in cols else None,
        birth_date="Date Of Birth" if "Date Of Birth" in cols else None,
        gender="Gender" if "Gender" in cols else None,
        # IMPORTANT UPDATE: Person Created At is the start date (tenure)
        signup_date="Person Created At" if "Person Created At" in cols else None,
        frequency_date="Frequency Date" if "Frequency Date" in cols else None,
        had_problem="Had Problem" if "Had Problem" in cols else None,
        solved_problem="Solved Problem" if "Solved Problem" in cols else None,
        problem_eval="Evaluation Problem" if "Evaluation Problem" in cols else None,
        problem_comment="Comment Problem" if "Comment Problem" in cols else None,
        touchpoint_eval_cols=touch_eval,
        touchpoint_comment_cols=touch_comment,
        tag_cols=tag_cols,
    )


# -----------------------------
# Preparation
# -----------------------------

def prepare_dataframe(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    data = df.copy()

    data[schema.nps] = pd.to_numeric(data[schema.nps], errors="coerce")
    data["nps_class"] = data[schema.nps].apply(classify_nps)

    if schema.birth_date:
        data["birth_dt"] = to_datetime(data[schema.birth_date])
        age = ((pd.Timestamp.today().normalize() - data["birth_dt"]).dt.days / 365.25)
        data["age"] = age.round(1)
        data["age_band"] = data["age"].apply(age_band)
    else:
        data["age"] = np.nan
        data["age_band"] = "Sem idade"

    if schema.signup_date:
        data["signup_dt"] = to_datetime(data[schema.signup_date])
        tenure = ((pd.Timestamp.today().normalize() - data["signup_dt"]).dt.days / 30.44)
        data["tenure_months"] = tenure.round(1)
        data["tenure_band"] = data["tenure_months"].apply(tenure_band)
    else:
        data["tenure_months"] = np.nan
        data["tenure_band"] = "Sem contrato"

    if schema.frequency_date:
        data["frequency_dt"] = to_datetime(data[schema.frequency_date])
        data["period"] = data["frequency_dt"].apply(day_period)
        data["week"] = data["frequency_dt"].dt.to_period("W").astype(str)
    else:
        data["frequency_dt"] = pd.NaT
        data["period"] = "Sem horário"
        data["week"] = "Sem semana"

    if schema.gender:
        data["gender_clean"] = data[schema.gender].fillna("Sem gênero").astype(str).str.strip()
    else:
        data["gender_clean"] = "Sem gênero"

    unit_col = schema.response_unit or schema.unit
    data["unit_clean"] = data[unit_col].fillna(data[schema.unit]).fillna("Sem unidade").astype(str).str.strip()

    if schema.had_problem:
        data["had_problem_clean"] = data[schema.had_problem].apply(safe_bool_text)
    else:
        data["had_problem_clean"] = "Não informado"

    if schema.solved_problem:
        data["solved_problem_clean"] = data[schema.solved_problem].apply(safe_bool_text)
    else:
        data["solved_problem_clean"] = "Não informado"

    return data


# -----------------------------
# Metrics
# -----------------------------

def segment_nps(data: pd.DataFrame, segment_col: str) -> pd.DataFrame:
    summary = (
        data.groupby(segment_col, dropna=False)["Evaluation"]
        .agg(respostas="count", nps=nps_score)
        .reset_index()
        .sort_values("nps", ascending=False)
    )
    return summary


def touchpoint_summary(data: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    rows = []
    for col in schema.touchpoint_eval_cols:
        name = col.replace(" Evaluation", "")
        vals = pd.to_numeric(data[col], errors="coerce")
        valid = vals.notna()
        row = {
            "touchpoint": name,
            "respostas": int(valid.sum()),
            "media": round(vals.mean(), 2) if valid.any() else np.nan,
            "nps_touchpoint": round(nps_score(vals), 1) if valid.any() else np.nan,
            "odds_ratio": round(odds_ratio_topbox(data, col), 2) if valid.any() else np.nan,
        }
        if pd.notna(row["nps_touchpoint"]) and pd.notna(row["odds_ratio"]):
            # Lower NPS + higher OR = higher priority
            row["priority_index"] = round((100 - row["nps_touchpoint"]) * np.log1p(row["odds_ratio"]), 2)
        else:
            row["priority_index"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["priority_index", "odds_ratio"], ascending=[False, False])


def problem_summary(data: pd.DataFrame) -> Dict[str, float]:
    problem_yes = data[data["had_problem_clean"].str.lower() == "sim"]
    problem_no = data[data["had_problem_clean"].str.lower() == "não"]
    return {
        "pct_problem": round((len(problem_yes) / len(data)) * 100, 1) if len(data) else np.nan,
        "nps_problem": round(nps_score(problem_yes["Evaluation"]), 1) if len(problem_yes) else np.nan,
        "nps_no_problem": round(nps_score(problem_no["Evaluation"]), 1) if len(problem_no) else np.nan,
        "problem_count": int(len(problem_yes)),
    }


def tag_summary(data: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    rows = []
    for col in schema.tag_cols:
        non_null = data[col].dropna()
        if non_null.empty:
            continue
        for item in non_null.astype(str):
            for kind, phrase in extract_type_and_phrases(item):
                rows.append({"tag": col, "kind": kind, "phrase": phrase})
    if not rows:
        return pd.DataFrame(columns=["tag", "kind", "count"])
    tags = pd.DataFrame(rows)
    summary = tags.groupby(["tag", "kind"]).size().reset_index(name="count").sort_values("count", ascending=False)
    return summary


def top_comment_themes(data: pd.DataFrame, schema: Schema, n: int = 10) -> pd.DataFrame:
    rows = []
    all_text_cols = [c for c in [schema.nps_comment, schema.problem_comment] if c]
    all_text_cols += schema.touchpoint_comment_cols

    for col in all_text_cols:
        if col not in data.columns:
            continue
        col_name = col.replace(" Comment", "")
        subset = data[["nps_class", col]].dropna()
        for _, row in subset.iterrows():
            text = normalize_text(row[col])
            if text:
                rows.append({"source": col_name, "nps_class": row["nps_class"], "text": text})

    if not rows:
        return pd.DataFrame(columns=["source", "nps_class", "count"])

    comments = pd.DataFrame(rows)
    comments["source"] = comments["source"].str.strip()
    summary = comments.groupby(["source", "nps_class"]).size().reset_index(name="count").sort_values("count", ascending=False)
    return summary.head(n)


# -----------------------------
# Insight engine
# -----------------------------

def build_insights(data: pd.DataFrame, schema: Schema, benchmark: float) -> Dict[str, object]:
    overall_nps = round(nps_score(data[schema.nps]), 1)
    zone = nps_zone(overall_nps)

    touch = touchpoint_summary(data, schema)
    tags = tag_summary(data, schema)
    problems = problem_summary(data)
    period = segment_nps(data, "period")
    units = segment_nps(data, "unit_clean").sort_values(["respostas", "nps"], ascending=[False, True])
    weekly = segment_nps(data, "week").sort_values("week")

    top_priorities = touch.head(3).copy()
    top_strengths = touch.sort_values(["nps_touchpoint", "odds_ratio"], ascending=[False, False]).head(3).copy()

    complaint_tags = tags[tags["kind"] == "COMPLAINT"].head(10)
    compliment_tags = tags[tags["kind"] == "COMPLIMENT"].head(10)
    suggestion_tags = tags[tags["kind"] == "SUGGESTION"].head(10)

    why_fell = None
    if len(weekly) >= 2:
        recent = weekly.tail(2).copy()
        recent_values = recent["nps"].tolist()
        if len(recent_values) == 2 and all(pd.notna(recent_values)):
            why_fell = recent_values[-1] - recent_values[-2]

    return {
        "overall_nps": overall_nps,
        "zone": zone,
        "benchmark_gap": round(overall_nps - benchmark, 1) if pd.notna(overall_nps) else np.nan,
        "touchpoints": touch,
        "problems": problems,
        "tags": tags,
        "complaint_tags": complaint_tags,
        "compliment_tags": compliment_tags,
        "suggestion_tags": suggestion_tags,
        "period": period,
        "units": units,
        "weekly": weekly,
        "top_priorities": top_priorities,
        "top_strengths": top_strengths,
        "why_fell_delta": why_fell,
    }


def format_bullets_from_df(df: pd.DataFrame, label_col: str, value_col: str, max_items: int = 3) -> List[str]:
    items = []
    for _, row in df.head(max_items).iterrows():
        items.append(f"- **{row[label_col]}**: {row[value_col]}")
    return items


def answer_question(question: str, insights: Dict[str, object]) -> str:
    q = question.lower().strip()
    overall_nps = insights["overall_nps"]
    zone = insights["zone"]
    benchmark_gap = insights["benchmark_gap"]
    priorities = insights["top_priorities"]
    strengths = insights["top_strengths"]
    problems = insights["problems"]
    complaint_tags = insights["complaint_tags"]
    compliment_tags = insights["compliment_tags"]
    weekly = insights["weekly"]
    delta = insights["why_fell_delta"]

    if any(x in q for x in ["maiores pontos de atenção", "pontos de atenção", "atenção"]):
        bullets = []
        for _, row in priorities.iterrows():
            bullets.append(
                f"- **{row['touchpoint']}** — NPS do ponto: **{row['nps_touchpoint']}**, odds ratio: **{row['odds_ratio']}**"
            )
        return (
            f"Seu NPS geral está em **{overall_nps}**, na **{zone}**, com gap de **{benchmark_gap}** pontos contra o benchmark.\n\n"
            f"Os maiores pontos de atenção hoje são:\n" + "\n".join(bullets)
        )

    if any(x in q for x in ["melhorar meu nps", "melhorar o nps", "como melhorar"]):
        actions = []
        for _, row in priorities.iterrows():
            actions.append(
                f"- Priorize **{row['touchpoint']}**, porque ele combina impacto alto no engajamento com desempenho ainda abaixo do ideal."
            )
        problem_text = ""
        if pd.notna(problems["nps_problem"]) and pd.notna(problems["nps_no_problem"]):
            problem_text = (
                f"\n\nQuando há problema reportado, o NPS fica em **{problems['nps_problem']}**, versus **{problems['nps_no_problem']}** entre quem não reportou problema."
            )
        return (
            f"Para melhorar o NPS, eu atacaria primeiro as maiores alavancas operacionais:\n"
            + "\n".join(actions)
            + problem_text
        )

    if any(x in q for x in ["maior diferencial", "diferencial", "pontos fortes"]):
        bullets = []
        for _, row in strengths.iterrows():
            bullets.append(
                f"- **{row['touchpoint']}** — NPS do ponto: **{row['nps_touchpoint']}**, odds ratio: **{row['odds_ratio']}**"
            )
        extra = ""
        if not compliment_tags.empty:
            top = compliment_tags.iloc[0]
            extra = f"\n\nNas tags positivas, o tema mais recorrente é **{top['tag']}**."
        return "Os maiores diferenciais percebidos pelo cliente são:\n" + "\n".join(bullets) + extra

    if any(x in q for x in ["3 maiores problemas", "três maiores problemas", "maiores problemas"]):
        bullets = []
        for _, row in priorities.head(3).iterrows():
            bullets.append(f"- **{row['touchpoint']}**")
        extra = ""
        if not complaint_tags.empty:
            tags_text = ", ".join(complaint_tags['tag'].head(3).tolist())
            extra = f"\n\nAs tags de reclamação mais frequentes reforçam esse diagnóstico: **{tags_text}**."
        return "Os 3 maiores problemas que você deveria atacar agora são:\n" + "\n".join(bullets) + extra

    if any(x in q for x in ["por que meu nps caiu", "porque meu nps caiu", "queda do nps"]):
        if len(weekly) < 2:
            return "Ainda não há períodos suficientes para explicar queda de NPS com segurança."
        last_weeks = weekly.tail(4).copy()
        weeks_lines = [f"- **{r['week']}**: {r['nps']}" for _, r in last_weeks.iterrows()]
        if delta is not None and pd.notna(delta) and delta < 0:
            return (
                "Houve uma queda recente no NPS.\n\n"
                "Evolução mais recente:\n" + "\n".join(weeks_lines) +
                f"\n\nA última variação foi de **{round(delta, 1)}** pontos. "
                "Os primeiros pontos para investigar são os touchpoints prioritários e as tags de reclamação mais recorrentes do período."
            )
        return "Não identifiquei queda recente clara no NPS.\n\n" + "\n".join(weeks_lines)

    return (
        f"Seu NPS geral está em **{overall_nps}** ({zone}). "
        "No MVP, eu respondo melhor perguntas como:\n"
        "- Quais são os maiores pontos de atenção?\n"
        "- O que fazer para melhorar meu NPS?\n"
        "- Qual é meu maior diferencial?\n"
        "- Quais são os 3 maiores problemas?\n"
        "- Por que meu NPS caiu?"
    )


# -----------------------------
# UI
# -----------------------------

def load_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def main():
    st.title("WeHelp Retention MVP")
    st.caption("MVP simples para leitura de planilha, cálculo de NPS, odds ratio e respostas guiadas de retenção.")

    with st.sidebar:
        st.header("Configurações")
        benchmark = st.number_input("Benchmark de NPS", min_value=-100.0, max_value=100.0, value=50.0, step=1.0)
        st.markdown("**Perguntas sugeridas**")
        st.markdown("- Quais são os maiores pontos de atenção?")
        st.markdown("- O que fazer para melhorar meu NPS?")
        st.markdown("- Qual é meu maior diferencial?")
        st.markdown("- Quais são os 3 maiores problemas?")
        st.markdown("- Por que meu NPS caiu?")

    uploaded_file = st.file_uploader("Suba a planilha limpa em XLSX ou CSV", type=["xlsx", "csv"])
    if not uploaded_file:
        st.info("Suba um arquivo para começar.")
        return

    raw = load_file(uploaded_file)
    schema = detect_schema(raw)
    data = prepare_dataframe(raw, schema)
    insights = build_insights(data, schema, benchmark)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("NPS Geral", insights["overall_nps"])
    col2.metric("Zona", insights["zone"])
    col3.metric("Gap vs Benchmark", insights["benchmark_gap"])
    col4.metric("Respostas", len(data))

    with st.expander("Schema detectado", expanded=False):
        st.write(schema)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Resumo executivo",
        "Segmentações",
        "Touchpoints",
        "Tags e comentários",
        "Perguntas ao agente",
    ])

    with tab1:
        st.subheader("Resumo executivo")
        st.markdown(answer_question("quais são os maiores pontos de atenção", insights))
        st.markdown("---")
        st.subheader("Top 3 prioridades")
        st.dataframe(insights["top_priorities"], use_container_width=True)
        st.subheader("Top 3 forças")
        st.dataframe(insights["top_strengths"], use_container_width=True)
        st.subheader("Problemas reportados")
        st.json(insights["problems"])

    with tab2:
        st.subheader("NPS por período")
        st.dataframe(insights["period"], use_container_width=True)
        st.subheader("NPS por unidade")
        st.dataframe(insights["units"], use_container_width=True)
        st.subheader("NPS por faixa etária")
        st.dataframe(segment_nps(data, "age_band"), use_container_width=True)
        st.subheader("NPS por tempo de contrato")
        st.dataframe(segment_nps(data, "tenure_band"), use_container_width=True)
        st.subheader("NPS por gênero")
        st.dataframe(segment_nps(data, "gender_clean"), use_container_width=True)
        st.subheader("Evolução por semana")
        st.dataframe(insights["weekly"], use_container_width=True)

    with tab3:
        st.subheader("Touchpoints")
        st.dataframe(insights["touchpoints"], use_container_width=True)

    with tab4:
        st.subheader("Resumo de tags")
        st.dataframe(insights["tags"], use_container_width=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Reclamações**")
            st.dataframe(insights["complaint_tags"], use_container_width=True)
        with c2:
            st.markdown("**Elogios**")
            st.dataframe(insights["compliment_tags"], use_container_width=True)
        with c3:
            st.markdown("**Sugestões**")
            st.dataframe(insights["suggestion_tags"], use_container_width=True)
        st.subheader("Fontes de comentário mais frequentes")
        st.dataframe(top_comment_themes(data, schema, n=20), use_container_width=True)

    with tab5:
        st.subheader("Pergunte ao agente")
        question = st.text_input("Digite sua pergunta", value="Quais são os maiores pontos de atenção?")
        if st.button("Responder"):
            st.markdown(answer_question(question, insights))


if __name__ == "__main__":
    main()
