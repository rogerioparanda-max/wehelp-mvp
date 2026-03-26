import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="WeHelp Retention MVP", layout="wide")


# =============================
# Helpers
# =============================

def normalize_text(value) -> str:
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
    return round(promoters - detractors, 1)


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

    a, b, c, d = [x + 0.5 for x in (a, b, c, d)]
    return round((a / b) / (c / d), 2)


def md_join(lines: List[str]) -> str:
    return "\n".join([str(x) for x in lines if str(x).strip()])


# =============================
# Schema
# =============================

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


def detect_schema(df: pd.DataFrame) -> Schema:
    cols = list(df.columns)
    touch_eval = [c for c in cols if str(c).endswith(" Evaluation")]
    touch_comment = [c for c in cols if str(c).endswith(" Comment")]

    known_non_tag = {
        "Internal Code", "Name", "Phone", "Email", "Document",
        "Person Created At", "Date Of Birth", "Gender",
        "Person Company Unit Name", "Response Company Unit Name",
        "Country Name", "State Name", "Evaluation", "Nps Comment",
        "Authorization", "Nps Status", "Had Problem", "Solved Problem",
        "Evaluation Problem", "Comment Problem", "Frequency Date", "Messages",
        "Data Fim Plano", "Data Inicio Plano", "Nome do Plano", "Professor",
        "Tipo do Plano", "Valor do Plano",
        "Para finalizar, como você avalia a temperatura na academia durante sua visita?",
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


# =============================
# Preparation
# =============================

def prepare_dataframe(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    data = df.copy()
    data[schema.nps] = pd.to_numeric(data[schema.nps], errors="coerce")
    data["nps_class"] = data[schema.nps].apply(classify_nps)

    if schema.birth_date:
        data["birth_dt"] = to_datetime(data[schema.birth_date])
        data["age"] = ((pd.Timestamp.today().normalize() - data["birth_dt"]).dt.days / 365.25).round(1)
        data["age_band"] = data["age"].apply(age_band)
    else:
        data["age"] = np.nan
        data["age_band"] = "Sem idade"

    if schema.signup_date:
        data["signup_dt"] = to_datetime(data[schema.signup_date])
        data["tenure_months"] = ((pd.Timestamp.today().normalize() - data["signup_dt"]).dt.days / 30.44).round(1)
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

    data["gender_clean"] = data[schema.gender].fillna("Sem gênero").astype(str).str.strip() if schema.gender else "Sem gênero"
    data["plan_type_clean"] = data[schema.plan_type].fillna("Sem tipo de plano").astype(str).str.strip() if schema.plan_type else "Sem tipo de plano"
    data["plan_name_clean"] = data[schema.plan_name].fillna("Sem nome de plano").astype(str).str.strip() if schema.plan_name else "Sem nome de plano"

    unit_col = schema.response_unit or schema.unit
    data["unit_clean"] = data[unit_col].fillna(data[schema.unit]).fillna("Sem unidade").astype(str).str.strip()

    data["had_problem_clean"] = data[schema.had_problem].apply(safe_bool_text) if schema.had_problem else "Não informado"
    data["solved_problem_clean"] = data[schema.solved_problem].apply(safe_bool_text) if schema.solved_problem else "Não informado"

    return data


# =============================
# Metrics
# =============================

def segment_nps(data: pd.DataFrame, segment_col: str) -> pd.DataFrame:
    return (
        data.groupby(segment_col, dropna=False)["Evaluation"]
        .agg(respostas="count", nps=nps_score)
        .reset_index()
        .sort_values("nps", ascending=False)
    )


def touchpoint_summary(data: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    rows = []
    for col in schema.touchpoint_eval_cols:
        vals = pd.to_numeric(data[col], errors="coerce")
        valid = vals.notna()
        if not valid.any():
            continue
        touch_nps = nps_score(vals)
        odds = odds_ratio_topbox(data, col)
        priority_index = round((100 - touch_nps) * np.log1p(odds), 2) if pd.notna(touch_nps) and pd.notna(odds) else np.nan
        rows.append({
            "touchpoint": col.replace(" Evaluation", ""),
            "respostas": int(valid.sum()),
            "media": round(vals.mean(), 2),
            "nps_touchpoint": touch_nps,
            "odds_ratio": odds,
            "priority_index": priority_index,
        })
    return pd.DataFrame(rows).sort_values(["priority_index", "odds_ratio"], ascending=[False, False]) if rows else pd.DataFrame()


def problem_summary(data: pd.DataFrame) -> Dict[str, float]:
    problem_yes = data[data["had_problem_clean"].str.lower() == "sim"]
    problem_no = data[data["had_problem_clean"].str.lower() == "não"]
    return {
        "pct_problem": round((len(problem_yes) / len(data)) * 100, 1) if len(data) else np.nan,
        "nps_problem": nps_score(problem_yes["Evaluation"]) if len(problem_yes) else np.nan,
        "nps_no_problem": nps_score(problem_no["Evaluation"]) if len(problem_no) else np.nan,
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
    return tags.groupby(["tag", "kind"]).size().reset_index(name="count").sort_values("count", ascending=False)


def collect_comment_evidence(data: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    rows = []
    text_cols = []
    if schema.nps_comment and schema.nps_comment in data.columns:
        text_cols.append((schema.nps_comment, "NPS geral"))
    if schema.problem_comment and schema.problem_comment in data.columns:
        text_cols.append((schema.problem_comment, "Problema reportado"))
    for col in schema.touchpoint_comment_cols:
        if col in data.columns:
            text_cols.append((col, col.replace(" Comment", "")))

    for col, source_name in text_cols:
        subset = data[["nps_class", col]].dropna()
        for _, row in subset.iterrows():
            text = normalize_text(row[col])
            if text:
                rows.append({"touchpoint": source_name, "nps_class": row["nps_class"], "comment": text})
    if not rows:
        return pd.DataFrame(columns=["touchpoint", "nps_class", "comment"])
    return pd.DataFrame(rows)


def find_comment_examples(comment_df: pd.DataFrame, touchpoint: Optional[str] = None, nps_class: Optional[str] = None, limit: int = 2) -> List[str]:
    if comment_df is None or comment_df.empty:
        return []
    temp = comment_df.copy()
    if touchpoint:
        temp = temp[temp["touchpoint"].astype(str).str.strip().str.lower() == str(touchpoint).strip().lower()]
    if nps_class:
        temp = temp[temp["nps_class"] == nps_class]
    if temp.empty:
        return []

    examples: List[str] = []
    seen = set()
    for text in temp["comment"].astype(str):
        clean = re.sub(r"\s+", " ", text).strip()
        if not clean:
            continue
        short = clean[:180] + ("..." if len(clean) > 180 else "")
        key = short.lower()
        if key not in seen:
            seen.add(key)
            examples.append(short)
        if len(examples) >= limit:
            break
    return examples


# =============================
# Insight engine
# =============================

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
            "mean_nps": round(mean_nps, 1),
            "std_nps": round(std_nps, 1) if pd.notna(std_nps) else np.nan,
            "mean_odds": odds,
            "best_unit": best_row["unit_clean"],
            "best_nps": round(best_row["nps_touchpoint"], 1),
            "worst_unit": worst_row["unit_clean"],
            "worst_nps": round(worst_row["nps_touchpoint"], 1),
            "variability_type": variability_type,
        })
    return pd.DataFrame(rows).sort_values(["std_nps", "mean_odds"], ascending=[False, False]) if rows else pd.DataFrame()


def build_insights(data: pd.DataFrame, schema: Schema, benchmark: float, selected_unit: str) -> Dict[str, object]:
    network_data = data.copy()
    unit_data = data[data["unit_clean"] == selected_unit].copy() if selected_unit != "Todas as unidades" else data.copy()

    overall_nps = nps_score(unit_data[schema.nps])
    network_nps = nps_score(network_data[schema.nps])
    zone = nps_zone(overall_nps)

    touch = touchpoint_summary(unit_data, schema)
    if not touch.empty:
        touch["bucket"] = touch.apply(lambda r: classify_touchpoint_bucket(r["nps_touchpoint"], r["odds_ratio"]), axis=1)

    tags = tag_summary(unit_data, schema)
    comments = collect_comment_evidence(unit_data, schema)
    problems = problem_summary(unit_data)
    period = segment_nps(unit_data, "period")
    units = segment_nps(network_data, "unit_clean")
    weekly = segment_nps(unit_data, "week").sort_values("week")
    variability = touchpoint_network_variability(network_data, schema)

    top_priorities = touch.sort_values(["priority_index", "odds_ratio"], ascending=[False, False]).head(3).copy() if not touch.empty else pd.DataFrame()
    top_strengths = touch.sort_values(["nps_touchpoint", "odds_ratio"], ascending=[False, False]).head(3).copy() if not touch.empty else pd.DataFrame()

    complaint_tags = tags[tags["kind"] == "COMPLAINT"].head(10)
    compliment_tags = tags[tags["kind"] == "COMPLIMENT"].head(10)
    suggestion_tags = tags[tags["kind"] == "SUGGESTION"].head(10)

    why_fell = None
    if len(weekly) >= 2:
        vals = weekly["nps"].tolist()
        if len(vals) >= 2 and pd.notna(vals[-1]) and pd.notna(vals[-2]):
            why_fell = vals[-1] - vals[-2]

    unit_vs_network_gap = round(overall_nps - network_nps, 1) if selected_unit != "Todas as unidades" else np.nan

    return {
        "selected_unit": selected_unit,
        "overall_nps": overall_nps,
        "zone": zone,
        "benchmark_gap": round(overall_nps - benchmark, 1) if pd.notna(overall_nps) else np.nan,
        "network_nps": network_nps,
        "unit_vs_network_gap": unit_vs_network_gap,
        "touchpoints": touch,
        "comment_evidence": comments,
        "network_variability": variability,
        "problems": problems,
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


# =============================
# Q&A
# =============================

def answer_question(question: str, insights: Dict[str, object]) -> str:
    q = normalize_text(question).lower()
    selected_unit = insights["selected_unit"]
    overall_nps = insights["overall_nps"]
    zone = insights["zone"]
    benchmark_gap = insights["benchmark_gap"]
    network_nps = insights["network_nps"]
    unit_vs_network_gap = insights["unit_vs_network_gap"]
    priorities = insights["top_priorities"]
    strengths = insights["top_strengths"]
    problems = insights["problems"]
    complaint_tags = insights["complaint_tags"]
    compliment_tags = insights["compliment_tags"]
    suggestion_tags = insights["suggestion_tags"]
    weekly = insights["weekly"]
    period = insights["period"]
    variability = insights["network_variability"]
    comment_evidence = insights["comment_evidence"]
    delta = insights["why_fell_delta"]

    def top_tags(df: pd.DataFrame, max_items: int = 3) -> str:
        if df is None or df.empty:
            return "sem recorrência suficiente nas tags"
        return ", ".join(df["tag"].astype(str).head(max_items).tolist())

    def variability_for_touchpoint(tp: str) -> Optional[pd.Series]:
        if variability is None or variability.empty:
            return None
        match = variability[variability["touchpoint"].astype(str).str.lower() == str(tp).lower()]
        if match.empty:
            return None
        return match.iloc[0]

    def executive_intro() -> str:
        parts = [
            f"**Leitura executiva da unidade**",
            f"Unidade analisada: **{selected_unit}**.",
            f"Seu NPS está em **{overall_nps}**, classificado como **{zone}**.",
            f"A distância para o benchmark é de **{benchmark_gap}** pontos.",
        ]
        if selected_unit != "Todas as unidades":
            parts.append(f"O NPS geral da rede está em **{network_nps}**.")
            if pd.notna(unit_vs_network_gap):
                direction = "acima" if unit_vs_network_gap >= 0 else "abaixo"
                parts.append(f"A sua unidade está **{abs(unit_vs_network_gap)}** pontos {direction} da média da rede.")
        return " ".join(parts)

    if any(x in q for x in ["maiores pontos de atenção", "pontos de atenção", "atenção"]):
        bullets = []
        for _, row in priorities.iterrows():
            bullets.append(
                f"- **{row['touchpoint']}**: NPS do ponto **{row['nps_touchpoint']}**, odds ratio **{row['odds_ratio']}** e classificação **{row['bucket']}**."
            )
        return md_join([
            executive_intro(),
            "",
            "**Onde estão os maiores pontos de atenção da sua unidade**",
            md_join(bullets) if bullets else "- Não há base suficiente para apontar prioridades com segurança.",
            "",
            "**Leitura de causa raiz**",
            f"As tags de reclamação mais recorrentes na sua unidade são **{top_tags(complaint_tags)}**.",
        ])

    if any(x in q for x in ["melhorar meu nps", "melhorar o nps", "como melhorar"]):
        if priorities.empty:
            return "Sem dados suficientes para análise."

        main_issue = priorities.iloc[0]["touchpoint"]
        examples = find_comment_examples(comment_evidence, touchpoint=main_issue, nps_class="Detrator", limit=2)
        evidence_text = md_join([f'- "{e}"' for e in examples]) if examples else "Sem comentários suficientes — mas o padrão quantitativo já é claro."

        strategic_line = (
            f"Você está **{abs(unit_vs_network_gap)}** pontos {'acima' if unit_vs_network_gap >= 0 else 'abaixo'} da rede."
            if pd.notna(unit_vs_network_gap)
            else "A leitura deve ser feita comparando unidade e rede."
        )

        var_row = variability_for_touchpoint(main_issue)
        variability_text = (
            f"Na rede, esse tema aparece como **{var_row['variability_type']}**."
            if var_row is not None
            else "Ainda sem leitura de variabilidade suficiente na rede."
        )

        return md_join([
            executive_intro(),
            "",
            "**Diagnóstico**",
            f"Você não tem vários problemas. O principal ponto que está puxando o NPS da sua unidade para baixo hoje é **{main_issue}**.",
            "",
            "**Por que isso acontece**",
            "Esse ponto aparece como prioridade porque combina baixo desempenho com alto poder de diferenciar promotores de detratores.",
            variability_text,
            "",
            "**O que você deve fazer hoje**",
            f"- Vá pessoalmente para **{main_issue}** e observe a operação ao vivo por pelo menos 30–60 minutos.",
            f"- Compare o que deveria acontecer em **{main_issue}** com o que realmente acontece.",
            f"- Identifique onde o padrão quebra e quem executa bem versus quem executa mal em **{main_issue}**.",
            "",
            "**O que ajustar com o time**",
            f"- Defina o padrão correto para **{main_issue}**.",
            f"- Mostre para o time o que não pode acontecer em **{main_issue}**.",
            f"- Reforce como o cliente percebe a falha nesse ponto.",
            "",
            "**Voz do cliente**",
            evidence_text,
            "",
            "**Leitura estratégica**",
            strategic_line,
            "Isso significa que sua unidade pode estar funcionando bem no geral, mas ainda com execução inconsistente nos pontos que mais pesam para retenção.",
            "",
            "**Resumo executivo**",
            f"Pare de tentar melhorar tudo. Se você corrigir **{main_issue}** com disciplina operacional, seu NPS sobe. Se não, você continua medindo o problema sem resolvê-lo.",
        ])

    if any(x in q for x in ["maior diferencial", "diferencial", "pontos fortes"]):
        bullets = []
        for _, row in strengths.iterrows():
            bullets.append(
                f"- **{row['touchpoint']}**: NPS do ponto **{row['nps_touchpoint']}**, odds ratio **{row['odds_ratio']}** e classificação **{row['bucket']}**."
            )
        positive_examples = find_comment_examples(comment_evidence, touchpoint=None, nps_class="Promotor", limit=2)
        return md_join([
            executive_intro(),
            "",
            "**Diferenciais percebidos na sua unidade**",
            md_join(bullets) if bullets else "- Não há base suficiente para apontar diferenciais com segurança.",
            "",
            "**Sinais qualitativos**",
            f"As tags positivas mais recorrentes são **{top_tags(compliment_tags)}**.",
            md_join([f'- "{e}"' for e in positive_examples]) if positive_examples else "",
        ])

    if any(x in q for x in ["3 maiores problemas", "três maiores problemas", "maiores problemas"]):
        bullets = []
        for _, row in priorities.iterrows():
            bullets.append(
                f"- **{row['touchpoint']}** — NPS do ponto **{row['nps_touchpoint']}**, odds ratio **{row['odds_ratio']}** e classificação **{row['bucket']}**."
            )
        return md_join([
            executive_intro(),
            "",
            "**Os 3 maiores problemas da sua unidade agora**",
            md_join(bullets) if bullets else "- Não há base suficiente para apontar os 3 maiores problemas com segurança.",
            "",
            "**Sinais dos comentários e tags**",
            f"As reclamações mais recorrentes são **{top_tags(complaint_tags)}**. As sugestões mais frequentes são **{top_tags(suggestion_tags)}**.",
        ])

    if any(x in q for x in ["por que meu nps caiu", "porque meu nps caiu", "queda do nps"]):
        if weekly is None or len(weekly) < 2:
            return md_join([executive_intro(), "", "Ainda não há períodos suficientes para explicar queda de NPS com segurança."])

        weeks_lines = [f"- **{r['week']}**: NPS **{r['nps']}** com **{int(r['respostas'])}** respostas" for _, r in weekly.tail(4).iterrows()]
        if pd.notna(delta) and delta < 0:
            return md_join([
                executive_intro(),
                "",
                f"**Queda recente identificada** Seu NPS caiu **{abs(round(delta, 1))}** pontos nos dois períodos mais recentes.",
                "",
                "**Evolução recente**",
                md_join(weeks_lines),
                "",
                "**Hipótese principal**",
                f"A queda precisa ser investigada a partir dos touchpoints críticos e das reclamações mais recorrentes, principalmente **{top_tags(complaint_tags)}**.",
            ])
        return md_join([executive_intro(), "", "**Evolução recente**", md_join(weeks_lines)])

    return md_join([
        executive_intro(),
        "",
        "**Perguntas que o MVP responde melhor**",
        "- Quais são os maiores pontos de atenção da minha unidade?",
        "- O que fazer para melhorar o NPS da minha unidade?",
        "- Qual é o maior diferencial da minha unidade?",
        "- Quais são os 3 maiores problemas da minha unidade?",
        "- Por que o NPS da minha unidade caiu?",
    ])


# =============================
# UI
# =============================

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

    unit_options = ["Todas as unidades"] + sorted([u for u in data["unit_clean"].dropna().unique().tolist() if str(u).strip()])
    selected_unit = st.selectbox("Selecione a unidade do gerente", options=unit_options, index=0)

    insights = build_insights(data, schema, benchmark, selected_unit=selected_unit)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("NPS da Unidade", insights["overall_nps"])
    col2.metric("Zona", insights["zone"])
    col3.metric("Gap vs Benchmark", insights["benchmark_gap"])
    if selected_unit != "Todas as unidades" and pd.notna(insights["unit_vs_network_gap"]):
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
            st.markdown(f"**NPS da unidade:** {insights['overall_nps']}")
            st.markdown(f"**NPS da rede:** {insights['network_nps']}")
            st.markdown(f"**Gap da unidade vs rede:** {insights['unit_vs_network_gap']}")
            st.subheader("Leitura de variabilidade por touchpoint na rede")
            st.dataframe(insights["network_variability"], use_container_width=True)

    with tab5:
        st.subheader("Resumo de tags")
        tags = tag_summary(data[data["unit_clean"] == selected_unit] if selected_unit != "Todas as unidades" else data, schema)
        st.dataframe(tags, use_container_width=True)
        st.subheader("Fontes de comentário mais frequentes")
        st.dataframe(top_comment_themes(data[data["unit_clean"] == selected_unit] if selected_unit != "Todas as unidades" else data, schema, n=20), use_container_width=True)

    with tab6:
        st.subheader("Pergunte ao agente")
        q = st.text_input("Digite sua pergunta", value="O que fazer para melhorar meu NPS?")
        if st.button("Responder"):
            st.markdown(answer_question(q, insights))


if __name__ == "__main__":
    main()
