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
    plan_type: Optional[str]
    plan_name: Optional[str]
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
        plan_type="Tipo do Plano" if "Tipo do Plano" in cols else None,
        plan_name="Nome do Plano" if "Nome do Plano" in cols else None,
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

    if schema.plan_type:
        data["plan_type_clean"] = data[schema.plan_type].fillna("Sem tipo de plano").astype(str).str.strip()
    else:
        data["plan_type_clean"] = "Sem tipo de plano"

    if schema.plan_name:
        data["plan_name_clean"] = data[schema.plan_name].fillna("Sem nome de plano").astype(str).str.strip()
    else:
        data["plan_name_clean"] = "Sem nome de plano"

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
    suggestion_tags = insights["suggestion_tags"]
    weekly = insights["weekly"]
    period = insights["period"]
    units = insights["units"]
    touchpoints = insights["touchpoints"]
    delta = insights["why_fell_delta"]

    def top_tags(df: pd.DataFrame, max_items: int = 3) -> str:
        if df is None or df.empty:
            return "sem recorrência suficiente nas tags"
        return ", ".join(df["tag"].astype(str).head(max_items).tolist())

    def worst_segment(df: pd.DataFrame, label_col: str) -> Optional[Tuple[str, float, int]]:
        valid = df[df["respostas"] >= 5].copy()
        valid = valid[pd.notna(valid["nps"])]
        if valid.empty:
            return None
        row = valid.sort_values("nps", ascending=True).iloc[0]
        return row[label_col], row["nps"], row["respostas"]

    def best_segment(df: pd.DataFrame, label_col: str) -> Optional[Tuple[str, float, int]]:
        valid = df[df["respostas"] >= 5].copy()
        valid = valid[pd.notna(valid["nps"])]
        if valid.empty:
            return None
        row = valid.sort_values("nps", ascending=False).iloc[0]
        return row[label_col], row["nps"], row["respostas"]

    if any(x in q for x in ["maiores pontos de atenção", "pontos de atenção", "atenção"]):
        bullets = []
        for _, row in priorities.head(3).iterrows():
            bullets.append(
                f"- **{row['touchpoint']}**: NPS do ponto **{row['nps_touchpoint']}**, odds ratio **{row['odds_ratio']}** e índice de prioridade **{row['priority_index']}**."
            )

        worst_period = worst_segment(period, "period")
        worst_unit = worst_segment(units, "unit_clean")
        evidence = []
        if worst_period:
            evidence.append(f"o pior período é **{worst_period[0]}** com NPS **{worst_period[1]}**")
        if worst_unit:
            evidence.append(f"a unidade com pior desempenho, entre as com base mínima, é **{worst_unit[0]}** com NPS **{worst_unit[1]}**")
        evidence_text = "; ".join(evidence) if evidence else "não há segmentação suficiente para apontar período ou unidade crítica com segurança"

        problem_text = ""
        if pd.notna(problems["nps_problem"]) and pd.notna(problems["nps_no_problem"]):
            problem_text = (
                f"

**Impacto de problema reportado**
"
                f"- NPS com problema: **{problems['nps_problem']}**
"
                f"- NPS sem problema: **{problems['nps_no_problem']}**
"
                f"- Clientes com problema reportado: **{problems['pct_problem']}%**"
            )

        return (
            f"**Diagnóstico geral**
"
            f"Seu NPS está em **{overall_nps}**, na **{zone}**, com diferença de **{benchmark_gap}** pontos versus o benchmark.

"
            f"**Maiores pontos de atenção**
"
            + "
".join(bullets)
            + f"

**Leitura executiva**
{evidence_text}.

"
            f"**Causa raiz mais provável**
As tags de reclamação mais recorrentes hoje são: **{top_tags(complaint_tags)}**."
            + problem_text
            + "

**O que isso significa**
Seu principal desafio não parece ser percepção geral da marca, e sim fricções operacionais concentradas em alavancas que influenciam diretamente a formação de promotores e detratores."
        )

    if any(x in q for x in ["melhorar meu nps", "melhorar o nps", "como melhorar"]):
        actions = []
        for _, row in priorities.head(3).iterrows():
            actions.append(
                f"- Ataque **{row['touchpoint']}** primeiro: ele combina desempenho insuficiente com alta capacidade de influenciar promotores versus detratores."
            )

        worst_period = worst_segment(period, "period")
        worst_unit = worst_segment(units, "unit_clean")
        focused_actions = []
        if worst_period:
            focused_actions.append(f"- Trate o período **{worst_period[0]}** como frente prioritária, porque é onde o NPS está mais pressionado.")
        if worst_unit:
            focused_actions.append(f"- Faça plano de ação específico para a unidade **{worst_unit[0]}**, que hoje aparece como o maior ponto de atenção operacional.")
        if pd.notna(problems["nps_problem"]) and pd.notna(problems["nps_no_problem"]):
            focused_actions.append(
                f"- Reforce o processo de resolução de problemas: clientes que reportam problema têm NPS **{problems['nps_problem']}**, versus **{problems['nps_no_problem']}** entre os que não reportam."
            )

        return (
            f"**Diagnóstico**
Seu NPS atual é **{overall_nps}** e está na **{zone}**. Para subir esse indicador, eu priorizaria ações em três níveis.

"
            f"**1. Alavancas estruturais**
" + "
".join(actions) +
            f"

**2. Foco operacional**
" + ("
".join(focused_actions) if focused_actions else "- Ainda não há segmentação suficiente para apontar onde concentrar a execução.") +
            f"

**3. Escuta qualitativa**
Use as reclamações mais recorrentes para direcionar os planos de ação. Hoje os temas mais citados são **{top_tags(complaint_tags)}**, enquanto os elogios mais recorrentes são **{top_tags(compliment_tags)}**.

"
            f"**Recomendação final**
Não tente melhorar tudo ao mesmo tempo. Escolha os 2 ou 3 touchpoints com maior prioridade, concentre a execução neles e acompanhe a mudança por unidade, período e tipo de plano."
        )

    if any(x in q for x in ["maior diferencial", "diferencial", "pontos fortes"]):
        bullets = []
        for _, row in strengths.head(3).iterrows():
            bullets.append(
                f"- **{row['touchpoint']}**: NPS do ponto **{row['nps_touchpoint']}** e odds ratio **{row['odds_ratio']}**."
            )
        best_unit = best_segment(units, "unit_clean")
        best_period = best_segment(period, "period")
        extra = []
        if best_unit:
            extra.append(f"a unidade destaque é **{best_unit[0]}** com NPS **{best_unit[1]}**")
        if best_period:
            extra.append(f"o melhor período é **{best_period[0]}** com NPS **{best_period[1]}**")
        extra_text = "; ".join(extra) if extra else "não há segmentação suficiente para apontar destaques operacionais"

        return (
            f"**Diagnóstico**
Seu maior diferencial hoje está menos na média geral e mais em alguns atributos da experiência que realmente ajudam a formar promotores.

"
            f"**Principais diferenciais percebidos**
" + "
".join(bullets) +
            f"

**Sinais qualitativos**
As tags positivas mais recorrentes são: **{top_tags(compliment_tags)}**.

"
            f"**Onde isso aparece com mais força**
{extra_text}.

"
            f"**Leitura executiva**
O seu diferencial competitivo percebido está nos pontos em que a experiência combina boa avaliação com forte influência sobre a probabilidade de o cliente virar promotor."
        )

    if any(x in q for x in ["3 maiores problemas", "três maiores problemas", "maiores problemas"]):
        bullets = []
        for _, row in priorities.head(3).iterrows():
            bullets.append(
                f"- **{row['touchpoint']}** — NPS do ponto **{row['nps_touchpoint']}**, odds ratio **{row['odds_ratio']}**."
            )
        return (
            f"**Os 3 maiores problemas para resolver agora**
" + "
".join(bullets) +
            f"

**Por que estes 3?**
Eles unem duas coisas ao mesmo tempo: desempenho abaixo do ideal e forte influência na formação de promotores e detratores.

"
            f"**Sinais dos comentários e tags**
As reclamações mais recorrentes hoje são: **{top_tags(complaint_tags)}**. As sugestões mais frequentes são: **{top_tags(suggestion_tags)}**.

"
            f"**Recomendação prática**
Monte planos de ação com dono, prazo e acompanhamento semanal para esses 3 temas antes de dispersar energia em temas secundários."
        )

    if any(x in q for x in ["por que meu nps caiu", "porque meu nps caiu", "queda do nps"]):
        if len(weekly) < 2:
            return "Ainda não há períodos suficientes para explicar queda de NPS com segurança."

        last_weeks = weekly.tail(4).copy()
        weeks_lines = [f"- **{r['week']}**: NPS **{r['nps']}** com **{int(r['respostas'])}** respostas" for _, r in last_weeks.iterrows()]

        touch_text = []
        for _, row in priorities.head(3).iterrows():
            touch_text.append(f"- **{row['touchpoint']}** continua entre os touchpoints mais sensíveis para explicar piora de percepção.")

        if delta is not None and pd.notna(delta) and delta < 0:
            return (
                f"**Queda recente identificada**
"
                f"Seu NPS caiu **{abs(round(delta, 1))}** pontos na comparação entre os dois períodos mais recentes.

"
                f"**Evolução recente**
" + "
".join(weeks_lines) +
                f"

**Hipótese principal**
A queda não deve ser lida só como oscilação estatística. Ela precisa ser investigada a partir dos touchpoints mais críticos e das reclamações mais recorrentes, principalmente: **{top_tags(complaint_tags)}**.

"
                f"**Onde olhar primeiro**
" + "
".join(touch_text) +
                f"

**Próximo passo recomendado**
Comparar a última semana com a anterior por unidade, período, tipo de plano e nome do plano para localizar exatamente onde a queda se concentrou."
            )

        return (
            f"**Leitura temporal**
Não identifiquei uma queda recente clara no NPS.

"
            f"**Evolução recente**
" + "
".join(weeks_lines)
        )

    return (
        f"**Resumo executivo**
Seu NPS geral está em **{overall_nps}** e a classificação atual é **{zone}**.

"
        f"**Leitura inicial**
Os principais temas de reclamação são **{top_tags(complaint_tags)}**, enquanto os temas positivos mais recorrentes são **{top_tags(compliment_tags)}**.

"
        f"**Perguntas que o MVP já responde melhor**
"
        f"- Quais são os maiores pontos de atenção?
"
        f"- O que fazer para melhorar meu NPS?
"
        f"- Qual é meu maior diferencial?
"
        f"- Quais são os 3 maiores problemas?
"
        f"- Por que meu NPS caiu?"
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
        st.subheader("NPS por tipo de plano")
        st.dataframe(segment_nps(data, "plan_type_clean"), use_container_width=True)
        st.subheader("NPS por nome do plano")
        st.dataframe(segment_nps(data, "plan_name_clean"), use_container_width=True)
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
