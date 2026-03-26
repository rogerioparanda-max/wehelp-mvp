import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="WeHelp Retention MVP", layout="wide")

# MVP note:
# The primary user is a unit manager. The app should answer from the selected unit's perspective
# and compare that unit against the overall network whenever possible.


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

def classify_touchpoint_bucket(nps_touchpoint: float, odds_ratio: float) -> str:
    if pd.isna(nps_touchpoint) or pd.isna(odds_ratio):
        return "Sem classificação"
    if odds_ratio >= 1.5 and nps_touchpoint < 50:
        return "Crítico"
    if odds_ratio >= 1.5 and nps_touchpoint >= 50:
        return "Diferencial"
    if odds_ratio >= 1.2:
        return "Oportunidade"
    return "Baixo impacto"


def touchpoint_network_variability(data: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    rows = []
    if "unit_clean" not in data.columns:
        return pd.DataFrame(columns=["touchpoint", "units", "mean_nps", "std_nps", "mean_odds", "best_unit", "worst_unit", "variability_type"])

    for col in schema.touchpoint_eval_cols:
        tp = col.replace(" Evaluation", "")
        per_unit = (
            data.groupby("unit_clean")[col]
            .apply(lambda s: nps_score(pd.to_numeric(s, errors="coerce")))
            .reset_index(name="nps_touchpoint")
        )
        per_unit = per_unit[pd.notna(per_unit["nps_touchpoint"])]
        if per_unit.empty:
            continue
        odds = odds_ratio_topbox(data, col)
        std_nps = per_unit["nps_touchpoint"].std()
        mean_nps = per_unit["nps_touchpoint"].mean()
        best_row = per_unit.sort_values("nps_touchpoint", ascending=False).iloc[0]
        worst_row = per_unit.sort_values("nps_touchpoint", ascending=True).iloc[0]
        if pd.notna(std_nps) and std_nps > 15:
            variability_type = "Problema de gestão"
        elif pd.notna(mean_nps) and mean_nps < 50 and pd.notna(odds) and odds >= 1.5:
            variability_type = "Problema estrutural"
        elif pd.notna(mean_nps) and mean_nps >= 60 and pd.notna(odds) and odds >= 1.5:
            variability_type = "Diferencial replicável"
        else:
            variability_type = "Neutro"
        rows.append({
            "touchpoint": tp,
            "units": int(len(per_unit)),
            "mean_nps": round(mean_nps, 1) if pd.notna(mean_nps) else np.nan,
            "std_nps": round(std_nps, 1) if pd.notna(std_nps) else np.nan,
            "mean_odds": round(odds, 2) if pd.notna(odds) else np.nan,
            "best_unit": best_row["unit_clean"],
            "best_nps": round(best_row["nps_touchpoint"], 1),
            "worst_unit": worst_row["unit_clean"],
            "worst_nps": round(worst_row["nps_touchpoint"], 1),
            "variability_type": variability_type,
        })
    return pd.DataFrame(rows).sort_values(["std_nps", "mean_odds"], ascending=[False, False]) if rows else pd.DataFrame(columns=["touchpoint", "units", "mean_nps", "std_nps", "mean_odds", "best_unit", "worst_unit", "variability_type"])


def build_insights(data: pd.DataFrame, schema: Schema, benchmark: float, selected_unit: Optional[str] = None) -> Dict[str, object]:
    network_data = data.copy()
    if selected_unit and selected_unit != "Todas as unidades":
        unit_data = data[data["unit_clean"] == selected_unit].copy()
    else:
        unit_data = data.copy()
        selected_unit = "Todas as unidades"

    overall_nps = round(nps_score(unit_data[schema.nps]), 1)
    zone = nps_zone(overall_nps)
    network_nps = round(nps_score(network_data[schema.nps]), 1)

    touch = touchpoint_summary(unit_data, schema)
    if not touch.empty:
        touch["bucket"] = touch.apply(lambda r: classify_touchpoint_bucket(r["nps_touchpoint"], r["odds_ratio"]), axis=1)

    tags = tag_summary(unit_data, schema)
    comment_evidence = collect_comment_evidence(unit_data, schema)
    problems = problem_summary(unit_data)
    period = segment_nps(unit_data, "period")
    units = segment_nps(network_data, "unit_clean").sort_values(["respostas", "nps"], ascending=[False, True])
    weekly = segment_nps(unit_data, "week").sort_values("week")
    variability = touchpoint_network_variability(network_data, schema)

    top_priorities = touch.sort_values(["priority_index", "odds_ratio"], ascending=[False, False]).head(3).copy() if not touch.empty else pd.DataFrame()
    top_strengths = touch.sort_values(["nps_touchpoint", "odds_ratio"], ascending=[False, False]).head(3).copy() if not touch.empty else pd.DataFrame()

    complaint_tags = tags[tags["kind"] == "COMPLAINT"].head(10)
    compliment_tags = tags[tags["kind"] == "COMPLIMENT"].head(10)
    suggestion_tags = tags[tags["kind"] == "SUGGESTION"].head(10)

    why_fell = None
    if len(weekly) >= 2:
        recent = weekly.tail(2).copy()
        recent_values = recent["nps"].tolist()
        if len(recent_values) == 2 and all(pd.notna(recent_values)):
            why_fell = recent_values[-1] - recent_values[-2]

    selected_unit_nps = overall_nps if selected_unit != "Todas as unidades" else np.nan
    unit_vs_network_gap = round(selected_unit_nps - network_nps, 1) if selected_unit != "Todas as unidades" and pd.notna(selected_unit_nps) and pd.notna(network_nps) else np.nan

    return {
        "selected_unit": selected_unit,
        "overall_nps": overall_nps,
        "zone": zone,
        "benchmark_gap": round(overall_nps - benchmark, 1) if pd.notna(overall_nps) else np.nan,
        "network_nps": network_nps,
        "unit_vs_network_gap": unit_vs_network_gap,
        "touchpoints": touch,
        "comment_evidence": comment_evidence,
        "network_variability": variability,
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
    selected_unit = insights.get("selected_unit", "Todas as unidades")
    overall_nps = insights["overall_nps"]
    zone = insights["zone"]
    benchmark_gap = insights["benchmark_gap"]
    network_nps = insights.get("network_nps", np.nan)
    unit_vs_network_gap = insights.get("unit_vs_network_gap", np.nan)
    priorities = insights["top_priorities"]
    strengths = insights["top_strengths"]
    problems = insights["problems"]
    complaint_tags = insights["complaint_tags"]
    compliment_tags = insights["compliment_tags"]
    suggestion_tags = insights["suggestion_tags"]
    weekly = insights["weekly"]
    period = insights["period"]
    units = insights["units"]
    variability = insights.get("network_variability", pd.DataFrame())
    comment_evidence = insights.get("comment_evidence", pd.DataFrame())
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

    def examples_for_touchpoint(touchpoint_name: str, nps_class: str = "Detrator", limit: int = 2) -> List[str]:
        examples = find_comment_examples(comment_evidence, touchpoint=touchpoint_name, nps_class=nps_class, limit=limit)
        if not examples:
            examples = find_comment_examples(comment_evidence, touchpoint=None, nps_class=nps_class, limit=limit)
        return examples

    def variability_for_touchpoint(tp: str) -> Optional[pd.Series]:
        if variability is None or variability.empty:
            return None
        match = variability[variability["touchpoint"].astype(str).str.lower() == str(tp).lower()]
        if match.empty:
            return None
        return match.iloc[0]

    def executive_intro() -> str:
        intro = f"""**Leitura executiva da unidade**
    Unidade analisada: **{selected_unit}**. Seu NPS está em **{overall_nps}**, classificado como **{zone}**.
    A distância para o benchmark é de **{benchmark_gap}** pontos."""
        
        if selected_unit != "Todas as unidades" and pd.notna(network_nps):
            intro += f" O NPS geral da rede está em **{network_nps}**"
            if pd.notna(unit_vs_network_gap):
                direction = "acima" if unit_vs_network_gap >= 0 else "abaixo"
                intro += f", e a sua unidade está **{abs(unit_vs_network_gap)}** pontos {direction} da média da rede."
            else:
                intro += "."
                
        return intro

    if any(x in q for x in ["maiores pontos de atenção", "pontos de atenção", "atenção"]):
        bullets = []
        evidence_parts = []
        network_parts = []
        for _, row in priorities.head(3).iterrows():
            tp = row["touchpoint"]
            bullets.append(
                f"- **{tp}**: NPS do ponto **{row['nps_touchpoint']}**, odds ratio **{row['odds_ratio']}** e classificação **{row.get('bucket', 'Sem classificação')}**."
            )
            examples = examples_for_touchpoint(tp, "Detrator", 1)
            if examples:
                evidence_parts.append(f"- Em **{tp}**, clientes da unidade dizem coisas como: \"{examples[0]}\"")
            var_row = variability_for_touchpoint(tp)
            if var_row is not None:
                network_parts.append(
                    f"- Na rede, **{tp}** se comporta como **{var_row['variability_type']}**. Melhor unidade: **{var_row['best_unit']}** ({var_row['best_nps']}); pior unidade: **{var_row['worst_unit']}** ({var_row['worst_nps']})."
                )

        worst_period = worst_segment(period, "period")
        evidence_text = f"o período mais pressionado da sua unidade é **{worst_period[0]}** com NPS **{worst_period[1]}**" if worst_period else "não há base suficiente para apontar um período crítico com segurança"

        problem_text = ""

        if pd.notna(problems["nps_problem"]) and pd.notna(problems["nps_no_problem"]):
            problem_text = f"""**Impacto do problema reportado na unidade**

        - NPS com problema: **{problems['nps_problem']}**
        - NPS sem problema: **{problems['nps_no_problem']}**
        - Clientes com problema reportado: **{problems['pct_problem']}%**
        """
            )

        return (
            executive_intro()
            + "

**Onde estão os maiores pontos de atenção da sua unidade**
"
            + ("
".join(bullets) if bullets else "- Não há base suficiente para apontar prioridades com segurança.")
            + f"""**Onde a sua operação está mais pressionada**
{evidence_text}."
            + f"""**Leitura de causa raiz**
As tags de reclamação mais recorrentes na sua unidade são **{top_tags(complaint_tags)}**."
            + ("

**Voz do cliente da unidade**
" + "
".join(evidence_parts) if evidence_parts else "")
            + ("

**Comparação com a rede**
" + "
".join(network_parts) if network_parts else "")
            + problem_text
            + "

**Conclusão**
O foco do gerente deve estar nos touchpoints que mais pressionam a sua unidade hoje, mas sempre comparando com a rede para entender o que é problema local de execução e o que é padrão estrutural."
        )

    if any(x in q for x in ["melhorar meu nps", "melhorar o nps", "como melhorar"]):
        actions = []
        evidence_parts = []
        network_parts = []
        for _, row in priorities.head(3).iterrows():
            tp = row["touchpoint"]
            actions.append(
                f"- **{tp}** deve entrar primeiro no plano de ação da sua unidade: ele combina desempenho insuficiente com alto poder de influenciar promotores versus detratores."
            )
            examples = examples_for_touchpoint(tp, "Detrator", 1)
            if examples:
                evidence_parts.append(f"- Em **{tp}**, os clientes da unidade relatam: \"{examples[0]}\"")
            var_row = variability_for_touchpoint(tp)
            if var_row is not None:
                network_parts.append(
                    f"- Na rede, **{tp}** está classificado como **{var_row['variability_type']}**. Isso ajuda a decidir se a ação deve ser local ou se o tema merece escalonamento para a rede."
                )

        worst_period = worst_segment(period, "period")
        focused_actions = []
        if worst_period:
            focused_actions.append(f"- Trate **{worst_period[0]}** como operação prioritária, porque é o período com pior NPS da sua unidade.")
        if pd.notna(problems["nps_problem"]) and pd.notna(problems["nps_no_problem"]):
            focused_actions.append(
                f"- Reforce o fechamento de problemas: na sua unidade, quem reporta problema tem NPS **{problems['nps_problem']}**, contra **{problems['nps_no_problem']}** entre os demais."
            )
        if pd.notna(unit_vs_network_gap) and unit_vs_network_gap < 0:
            focused_actions.append(
                f"- Sua unidade está abaixo da média da rede em **{abs(unit_vs_network_gap)}** pontos. Priorize execução disciplinada nos touchpoints críticos antes de abrir novas frentes."
            )

        return (
            executive_intro()
            + "

**O que fazer para melhorar o NPS da sua unidade**
"
            + ("
".join(actions) if actions else "- Não há base suficiente para definir prioridades com segurança.")
            + "

**Onde concentrar a execução do gerente**
"
            + ("
".join(focused_actions) if focused_actions else "- Ainda não há base suficiente para apontar um recorte operacional prioritário.")
            + f"

**Leitura qualitativa da unidade**
Os temas mais citados nas reclamações são **{top_tags(complaint_tags)}**, enquanto os elogios mais recorrentes são **{top_tags(compliment_tags)}**."
            + ("

**Voz do cliente da unidade**
" + "
".join(evidence_parts) if evidence_parts else "")
            + ("

**Comparação com a rede**
" + "
".join(network_parts) if network_parts else "")
            + "

**Recomendação executiva**
O gerente não deve tentar melhorar tudo ao mesmo tempo. O melhor caminho é atacar 2 ou 3 prioridades da unidade, medir semanalmente e comparar com o comportamento da rede para separar falha local de problema estrutural."
        )

    if any(x in q for x in ["maior diferencial", "diferencial", "pontos fortes"]):
        bullets = []
        network_parts = []
        for _, row in strengths.head(3).iterrows():
            tp = row["touchpoint"]
            bullets.append(
                f"- **{tp}**: NPS do ponto **{row['nps_touchpoint']}**, odds ratio **{row['odds_ratio']}** e classificação **{row.get('bucket', 'Sem classificação')}**."
            )
            var_row = variability_for_touchpoint(tp)
            if var_row is not None:
                network_parts.append(
                    f"- Na rede, **{tp}** aparece como **{var_row['variability_type']}**. Melhor unidade: **{var_row['best_unit']}** ({var_row['best_nps']})."
                )
        positive_examples = find_comment_examples(comment_evidence, touchpoint=None, nps_class="Promotor", limit=2)
        return (
            executive_intro()
            + "

**Diferenciais percebidos na sua unidade**
"
            + ("
".join(bullets) if bullets else "- Não há base suficiente para apontar diferenciais com segurança.")
            + f"

**Sinais qualitativos**
As tags positivas mais recorrentes na sua unidade são **{top_tags(compliment_tags)}**."
            + ("

**Evidências nos comentários**
" + "
".join([f'- "{e}"' for e in positive_examples]) if positive_examples else "")
            + ("

**Comparação com a rede**
" + "
".join(network_parts) if network_parts else "")
            + "

**Conclusão**
Os diferenciais da unidade devem ser protegidos e, quando também aparecem fortes na rede, podem ser tratados como boas práticas replicáveis."
        )

    if any(x in q for x in ["3 maiores problemas", "três maiores problemas", "maiores problemas"]):
        bullets = []
        evidence_parts = []
        network_parts = []
        for _, row in priorities.head(3).iterrows():
            tp = row["touchpoint"]
            bullets.append(
                f"- **{tp}** — NPS do ponto **{row['nps_touchpoint']}**, odds ratio **{row['odds_ratio']}** e classificação **{row.get('bucket', 'Sem classificação')}**."
            )
            examples = examples_for_touchpoint(tp, "Detrator", 1)
            if examples:
                evidence_parts.append(f"- Em **{tp}**, clientes da unidade descrevem situações como: \"{examples[0]}\"")
            var_row = variability_for_touchpoint(tp)
            if var_row is not None:
                network_parts.append(f"- **{tp}** na rede: **{var_row['variability_type']}**.")
        return (
            executive_intro()
            + "

**Os 3 maiores problemas da sua unidade agora**
"
            + ("
".join(bullets) if bullets else "- Não há base suficiente para apontar os 3 maiores problemas com segurança.")
            + f"

**Sinais dos comentários e tags**
As reclamações mais recorrentes são **{top_tags(complaint_tags)}**. As sugestões mais frequentes são **{top_tags(suggestion_tags)}**."
            + ("

**Evidências nos comentários**
" + "
".join(evidence_parts) if evidence_parts else "")
            + ("

**Comparação com a rede**
" + "
".join(network_parts) if network_parts else "")
            + "

**Recomendação executiva**
Monte plano de ação da unidade com dono, prazo e acompanhamento semanal para esses 3 temas. Onde a rede mostrar alta variabilidade, trate como disciplina de execução local; onde o padrão for estrutural, escale o tema."
        )

    if any(x in q for x in ["por que meu nps caiu", "porque meu nps caiu", "queda do nps"]):
        if len(weekly) < 2:
            return executive_intro() + "

Ainda não há períodos suficientes para explicar queda de NPS com segurança."
        last_weeks = weekly.tail(4).copy()
        weeks_lines = [f"- **{r['week']}**: NPS **{r['nps']}** com **{int(r['respostas'])}** respostas" for _, r in last_weeks.iterrows()]
        recent_examples = find_comment_examples(comment_evidence, touchpoint=None, nps_class="Detrator", limit=2)
        touch_text = []
        for _, row in priorities.head(3).iterrows():
            touch_text.append(f"- **{row['touchpoint']}** segue entre os touchpoints mais sensíveis para explicar piora de percepção na sua unidade.")
        if delta is not None and pd.notna(delta) and delta < 0:
            return (
                executive_intro()
                + f"

**Queda recente identificada na unidade**
Seu NPS caiu **{abs(round(delta, 1))}** pontos na comparação entre os dois períodos mais recentes."
                + "

**Evolução recente**
"
                + "
".join(weeks_lines)
                + f"

**Hipótese principal**
A queda da sua unidade não deve ser lida só como oscilação estatística. Ela precisa ser investigada a partir dos touchpoints mais críticos e das reclamações mais recorrentes, principalmente **{top_tags(complaint_tags)}**."
                + "

**Onde olhar primeiro**
"
                + "
".join(touch_text)
                + ("

**Evidências nos comentários**
" + "
".join([f'- "{e}"' for e in recent_examples]) if recent_examples else "")
                + "

**Comparação com a rede**
Compare a trajetória da sua unidade com a média da rede para entender se a queda é local ou parte de um padrão mais amplo."
            )
        return executive_intro() + "

**Leitura temporal**
Não identifiquei uma queda recente clara no NPS da sua unidade.

**Evolução recente**
" + "
".join(weeks_lines)

    return (
        executive_intro()
        + f"

**Leitura inicial da unidade**
Os principais temas de reclamação são **{top_tags(complaint_tags)}**, enquanto os temas positivos mais recorrentes são **{top_tags(compliment_tags)}**."
        + "

**Perguntas que o MVP já responde melhor para o gerente**
"
        + "- Quais são os maiores pontos de atenção da minha unidade?
"
        + "- O que fazer para melhorar o NPS da minha unidade?
"
        + "- Qual é o maior diferencial da minha unidade?
"
        + "- Quais são os 3 maiores problemas da minha unidade?
"
        + "- Por que o NPS da minha unidade caiu?"
    )


def load_file(uploaded_file) -> pd.DataFrame:(uploaded_file) -> pd.DataFrame:(uploaded_file) -> pd.DataFrame:(uploaded_file) -> pd.DataFrame:
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

    unit_options = ["Todas as unidades"] + sorted([u for u in data["unit_clean"].dropna().unique().tolist() if str(u).strip()])
    selected_unit = st.selectbox("Selecione a unidade do gerente", options=unit_options, index=0)

    insights = build_insights(data, schema, benchmark, selected_unit=selected_unit)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("NPS da Unidade", insights["overall_nps"])
    col2.metric("Zona", insights["zone"])
    col3.metric("Gap vs Benchmark", insights["benchmark_gap"])
    if selected_unit != "Todas as unidades" and pd.notna(insights.get("unit_vs_network_gap", np.nan)):
        col4.metric("Gap vs Rede", insights["unit_vs_network_gap"])
    else:
        col4.metric("Respostas", len(data) if selected_unit == "Todas as unidades" else len(data[data["unit_clean"] == selected_unit]))

    with st.expander("Schema detectado", expanded=False):
        st.write(schema)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Resumo executivo",
        "Segmentações",
        "Touchpoints",
        "Rede vs Unidade",
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
        st.subheader("Comparação rede vs unidade")
        if selected_unit == "Todas as unidades":
            st.info("Selecione uma unidade específica para ver o comparativo do gerente contra a rede.")
        else:
            st.markdown(f"**Unidade selecionada:** {selected_unit}")
            st.markdown(f"**NPS da unidade:** {insights['overall_nps']}  ")
            st.markdown(f"**NPS da rede:** {insights['network_nps']}  ")
            st.markdown(f"**Gap da unidade vs rede:** {insights['unit_vs_network_gap']}  ")
            st.subheader("Leitura de variabilidade por touchpoint na rede")
            st.dataframe(insights["network_variability"], use_container_width=True)

    with tab5:
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

    with tab6:
        st.subheader("Pergunte ao agente")
        question = st.text_input("Digite sua pergunta", value="Quais são os maiores pontos de atenção?")
        if st.button("Responder"):
            st.markdown(answer_question(question, insights))


if __name__ == "__main__":
    main()
