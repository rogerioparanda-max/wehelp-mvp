import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="WeHelp Retention MVP", layout="wide")


def normalize_text(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def classify_nps(score) -> str:
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

    data["gender_clean"] = data[schema.gender].fillna("Sem gênero").astype(str).str.strip() if schema.gender else "Sem gênero"
    data["plan_type_clean"] = data[schema.plan_type].fillna("Sem tipo de plano").astype(str).str.strip() if schema.plan_type else "Sem tipo de plano"
    data["plan_name_clean"] = data[schema.plan_name].fillna("Sem nome de plano").astype(str).str.strip() if schema.plan_name else "Sem nome de plano"

    unit_col = schema.response_unit or schema.unit
    data["unit_clean"] = data[unit_col].fillna(data[schema.unit]).fillna("Sem unidade").astype(str).str.strip()
    data["had_problem_clean"] = data[schema.had_problem].apply(safe_bool_text) if schema.had_problem else "Não informado"
    data["solved_problem_clean"] = data[schema.solved_problem].apply(safe_bool_text) if schema.solved_problem else "Não informado"
    return data


def segment_nps(data: pd.DataFrame, segment_col: str) -> pd.DataFrame:
    rows = []
    for key, grp in data.groupby(segment_col, dropna=False):
        rows.append({segment_col: key, "respostas": int(len(grp)), "nps": nps_score(grp["Evaluation"])})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=[segment_col, "respostas", "nps"])


def touchpoint_summary(data: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    rows = []
    for col in schema.touchpoint_eval_cols:
        name = col.replace(" Evaluation", "")
        vals = pd.to_numeric(data[col], errors="coerce")
        valid = vals.notna()
        if not valid.any():
            continue
        nps_tp = nps_score(vals)
        odds = odds_ratio_topbox(data, col)
        priority_index = ((100 - nps_tp) * np.log1p(odds)) if pd.notna(nps_tp) and pd.notna(odds) else np.nan
        rows.append({
            "touchpoint": name,
            "respostas": int(valid.sum()),
            "media": round(vals.mean(), 2),
            "nps_touchpoint": nps_tp,
            "odds_ratio": odds,
            "priority_index": round(priority_index, 2) if pd.notna(priority_index) else np.nan,
        })
    if not rows:
        return pd.DataFrame(columns=["touchpoint", "respostas", "media", "nps_touchpoint", "odds_ratio", "priority_index"])
    return pd.DataFrame(rows).sort_values(["priority_index", "odds_ratio"], ascending=[False, False], na_position="last")


def problem_summary(data: pd.DataFrame) -> Dict[str, float]:
    problem_yes = data[data["had_problem_clean"].str.lower() == "sim"]
    problem_no = data[data["had_problem_clean"].str.lower() == "não"]
    return {
        "pct_problem": round((len(problem_yes) / len(data)) * 100, 1) if len(data) else np.nan,
        "nps_problem": nps_score(problem_yes["Evaluation"]) if len(problem_yes) else np.nan,
        "nps_no_problem": nps_score(problem_no["Evaluation"]) if len(problem_no) else np.nan,
        "problem_count": int(len(problem_yes)),
    }


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
                rows.append({"source": source_name, "nps_class": row["nps_class"], "text": text})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["source", "nps_class", "text"])


def find_comment_examples(comment_df: pd.DataFrame, touchpoint: Optional[str] = None, nps_class: Optional[str] = None, limit: int = 2) -> List[str]:
    if comment_df is None or comment_df.empty:
        return []
    temp = comment_df.copy()
    if touchpoint:
        tp = str(touchpoint).strip().lower()
        temp = temp[temp["source"].astype(str).str.strip().str.lower() == tp]
    if nps_class:
        temp = temp[temp["nps_class"] == nps_class]
    if temp.empty:
        return []
    examples, seen = [], set()
    for text in temp["text"].astype(str):
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
        per_unit = data.groupby("unit_clean")[col].apply(lambda s: nps_score(pd.to_numeric(s, errors="coerce"))).reset_index(name="nps_touchpoint")
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
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def build_insights(data: pd.DataFrame, schema: Schema, benchmark: float, selected_unit: Optional[str] = None) -> Dict[str, object]:
    network_data = data.copy()
    unit_data = data[data["unit_clean"] == selected_unit].copy() if selected_unit and selected_unit != "Todas as unidades" else data.copy()
    selected_unit = selected_unit or "Todas as unidades"

    overall_nps = nps_score(unit_data[schema.nps])
    network_nps = nps_score(network_data[schema.nps])
    touch = touchpoint_summary(unit_data, schema)
    if not touch.empty:
        touch["bucket"] = touch.apply(lambda r: classify_touchpoint_bucket(r["nps_touchpoint"], r["odds_ratio"]), axis=1)

    weekly = segment_nps(unit_data, "week")
    why_fell = np.nan
    if len(weekly) >= 2:
        vals = weekly.tail(2)["nps"].tolist()
        if len(vals) == 2 and all(pd.notna(vals)):
            why_fell = vals[-1] - vals[-2]

    unit_vs_network_gap = round(overall_nps - network_nps, 1) if selected_unit != "Todas as unidades" and pd.notna(overall_nps) and pd.notna(network_nps) else np.nan
    tags = tag_summary(unit_data, schema)

    return {
        "selected_unit": selected_unit,
        "overall_nps": overall_nps,
        "zone": nps_zone(overall_nps),
        "benchmark_gap": round(overall_nps - benchmark, 1) if pd.notna(overall_nps) else np.nan,
        "network_nps": network_nps,
        "unit_vs_network_gap": unit_vs_network_gap,
        "touchpoints": touch,
        "comment_evidence": collect_comment_evidence(unit_data, schema),
        "network_variability": touchpoint_network_variability(network_data, schema),
        "problems": problem_summary(unit_data),
        "tags": tags,
        "complaint_tags": tags[tags["kind"] == "COMPLAINT"].head(10),
        "compliment_tags": tags[tags["kind"] == "COMPLIMENT"].head(10),
        "suggestion_tags": tags[tags["kind"] == "SUGGESTION"].head(10),
        "period": segment_nps(unit_data, "period"),
        "units": segment_nps(network_data, "unit_clean"),
        "weekly": weekly,
        "top_priorities": touch.head(3).copy() if not touch.empty else pd.DataFrame(),
        "top_strengths": touch.sort_values(["nps_touchpoint", "odds_ratio"], ascending=[False, False]).head(3).copy() if not touch.empty else pd.DataFrame(),
        "why_fell_delta": why_fell,
    }


def answer_question(question: str, insights: Dict[str, object]) -> str:
    q = question.lower().strip()
    selected_unit = insights.get("selected_unit", "Todas as unidades")
    overall_nps = insights.get("overall_nps", np.nan)
    zone = insights.get("zone", "Sem dados")
    benchmark_gap = insights.get("benchmark_gap", np.nan)
    network_nps = insights.get("network_nps", np.nan)
    unit_vs_network_gap = insights.get("unit_vs_network_gap", np.nan)
    priorities = insights.get("top_priorities", pd.DataFrame())
    strengths = insights.get("top_strengths", pd.DataFrame())
    problems = insights.get("problems", {})
    complaint_tags = insights.get("complaint_tags", pd.DataFrame())
    compliment_tags = insights.get("compliment_tags", pd.DataFrame())
    suggestion_tags = insights.get("suggestion_tags", pd.DataFrame())
    weekly = insights.get("weekly", pd.DataFrame())
    period = insights.get("period", pd.DataFrame())
    variability = insights.get("network_variability", pd.DataFrame())
    comment_evidence = insights.get("comment_evidence", pd.DataFrame())
    delta = insights.get("why_fell_delta", np.nan)

    def top_tags(df: pd.DataFrame, max_items: int = 3) -> str:
        if df is None or df.empty:
            return "sem recorrência suficiente nas tags"
        return ", ".join(df["tag"].astype(str).head(max_items).tolist())

    def worst_segment(df: pd.DataFrame, label_col: str):
        if df is None or df.empty:
            return None
        valid = df[df["respostas"] >= 5].copy()
        valid = valid[pd.notna(valid["nps"])]
        if valid.empty:
            return None
        row = valid.sort_values("nps", ascending=True).iloc[0]
        return row[label_col], row["nps"], row["respostas"]

    def variability_for_touchpoint(tp: str):
        if variability is None or variability.empty:
            return None
        match = variability[variability["touchpoint"].astype(str).str.lower() == str(tp).lower()]
        return match.iloc[0] if not match.empty else None

    def examples_for_touchpoint(tp: str, cls: str = "Detrator", limit: int = 1):
        ex = find_comment_examples(comment_evidence, touchpoint=tp, nps_class=cls, limit=limit)
        return ex if ex else find_comment_examples(comment_evidence, touchpoint=None, nps_class=cls, limit=limit)

    intro_parts = [
        "**Leitura executiva da unidade**",
        f"Unidade analisada: **{selected_unit}**.",
        f"Seu NPS está em **{overall_nps}**, classificado como **{zone}**.",
        f"A distância para o benchmark é de **{benchmark_gap}** pontos.",
    ]
    if selected_unit != "Todas as unidades" and pd.notna(network_nps):
        intro_parts.append(f"O NPS geral da rede está em **{network_nps}**.")
        if pd.notna(unit_vs_network_gap):
            direction = "acima" if unit_vs_network_gap >= 0 else "abaixo"
            intro_parts.append(f"A sua unidade está **{abs(unit_vs_network_gap)}** pontos {direction} da média da rede.")
    intro = " ".join(intro_parts)

    if "atenção" in q:
        bullets, evidence_parts, network_parts = [], [], []
        for _, row in priorities.head(3).iterrows():
            tp = row["touchpoint"]
            bullets.append(f"- **{tp}**: NPS do ponto **{row['nps_touchpoint']}**, odds ratio **{row['odds_ratio']}** e classificação **{row.get('bucket', 'Sem classificação')}**.")
            ex = examples_for_touchpoint(tp)
            if ex:
                evidence_parts.append(f'- Em **{tp}**, clientes da unidade dizem coisas como: "{ex[0]}".')
            var_row = variability_for_touchpoint(tp)
            if var_row is not None:
                network_parts.append(f"- Na rede, **{tp}** se comporta como **{var_row['variability_type']}**. Melhor unidade: **{var_row['best_unit']}** ({var_row['best_nps']}); pior unidade: **{var_row['worst_unit']}** ({var_row['worst_nps']}).")
        worst = worst_segment(period, "period")
        evidence_text = f"O período mais pressionado da sua unidade é **{worst[0]}** com NPS **{worst[1]}**." if worst else "Não há base suficiente para apontar um período crítico com segurança."
        lines = [intro, "", "**Onde estão os maiores pontos de atenção da sua unidade**", md_join(bullets) if bullets else "- Não há base suficiente para apontar prioridades com segurança.", "", "**Onde a sua operação está mais pressionada**", evidence_text, "", "**Leitura de causa raiz**", f"As tags de reclamação mais recorrentes na sua unidade são **{top_tags(complaint_tags)}**."]
        if evidence_parts:
            lines += ["", "**Voz do cliente da unidade**", md_join(evidence_parts)]
        if network_parts:
            lines += ["", "**Comparação com a rede**", md_join(network_parts)]
        if pd.notna(problems.get("nps_problem", np.nan)) and pd.notna(problems.get("nps_no_problem", np.nan)):
            lines += ["", "**Impacto do problema reportado na unidade**", f"- NPS com problema: **{problems['nps_problem']}**", f"- NPS sem problema: **{problems['nps_no_problem']}**", f"- Clientes com problema reportado: **{problems['pct_problem']}%**"]
        lines += ["", "**Conclusão**", "O foco do gerente deve estar nos touchpoints que mais pressionam a sua unidade hoje, mas sempre comparando com a rede para entender o que é problema local de execução e o que é padrão estrutural."]
        return md_join(lines)

    if any(x in q for x in ["melhorar meu nps", "melhorar o nps", "como melhorar"]):

    priorities_list = priorities.head(3)

    diagnosis = []
    decisions = []
    actions_today = []
    team_actions = []
    evidence_parts = []

    for i, (_, row) in enumerate(priorities_list.iterrows(), start=1):
        tp = row["touchpoint"]

        diagnosis.append(
            f"{i}. **{tp}** está pressionando o NPS da sua unidade (NPS {row['nps_touchpoint']} | alto impacto)."
        )

        decisions.append(
            f"{i}. **{tp} é prioridade imediata.** Não tratar isso primeiro dilui qualquer esforço de melhoria."
        )

        actions_today.append(
            f"- Hoje: vá para **{tp}**, observe operação ao vivo e identifique falhas reais de execução."
        )

        team_actions.append(
            f"- Esta semana: alinhe padrão com o time de **{tp}**, mostre exemplos reais e cobre execução consistente."
        )

        examples = find_comment_examples(comment_evidence, touchpoint=tp, nps_class="Detrator", limit=1)
        if examples:
            evidence_parts.append(f"- Em **{tp}**, clientes dizem: \"{examples[0]}\"")

    risk = ""
    if pd.notna(unit_vs_network_gap) and unit_vs_network_gap < 0:
        risk = f"Sua unidade está abaixo da rede. Se nada for feito, a tendência é perda de clientes e aumento de churn."

    return f"""{executive_intro()}

**Diagnóstico direto**

{chr(10).join(diagnosis)}

**Decisão gerencial**

{chr(10).join(decisions)}

**O que você deve fazer HOJE (ação do gerente)**

{chr(10).join(actions_today)}

**O que ajustar com o time (execução)**

{chr(10).join(team_actions)}

**Voz do cliente (evidência real)**

{chr(10).join(evidence_parts) if evidence_parts else "Ainda sem comentários suficientes, mas o padrão quantitativo já indica problema claro."}

**Risco de não agir**

{risk if risk else "Mesmo acima da média, ignorar esses pontos reduz sua vantagem competitiva."}

**Mensagem final (nível CEO)**

NPS não melhora com análise.  
Melhora quando o gerente muda o padrão de execução da unidade.

Se você atacar esses 2–3 pontos com disciplina,  
você melhora retenção. Se não, continua medindo problema.
"""    

    if "diferencial" in q or "pontos fortes" in q:
        bullets, network_parts = [], []
        for _, row in strengths.head(3).iterrows():
            tp = row["touchpoint"]
            bullets.append(f"- **{tp}**: NPS do ponto **{row['nps_touchpoint']}**, odds ratio **{row['odds_ratio']}** e classificação **{row.get('bucket', 'Sem classificação')}**.")
            var_row = variability_for_touchpoint(tp)
            if var_row is not None:
                network_parts.append(f"- Na rede, **{tp}** aparece como **{var_row['variability_type']}**. Melhor unidade: **{var_row['best_unit']}** ({var_row['best_nps']}).")
        positives = find_comment_examples(comment_evidence, touchpoint=None, nps_class="Promotor", limit=2)
        lines = [intro, "", "**Diferenciais percebidos na sua unidade**", md_join(bullets) if bullets else "- Não há base suficiente para apontar diferenciais com segurança.", "", "**Sinais qualitativos**", f"As tags positivas mais recorrentes na sua unidade são **{top_tags(compliment_tags)}**."]
        if positives:
            lines += ["", "**Evidências nos comentários**", md_join([f'- "{e}"' for e in positives])]
        if network_parts:
            lines += ["", "**Comparação com a rede**", md_join(network_parts)]
        lines += ["", "**Conclusão**", "Os diferenciais da unidade devem ser protegidos e, quando também aparecem fortes na rede, podem ser tratados como boas práticas replicáveis."]
        return md_join(lines)

    if "3 maiores problemas" in q or "três maiores problemas" in q or "maiores problemas" in q:
        bullets, evidence_parts, network_parts = [], [], []
        for _, row in priorities.head(3).iterrows():
            tp = row["touchpoint"]
            bullets.append(f"- **{tp}** — NPS do ponto **{row['nps_touchpoint']}**, odds ratio **{row['odds_ratio']}** e classificação **{row.get('bucket', 'Sem classificação')}**.")
            ex = examples_for_touchpoint(tp)
            if ex:
                evidence_parts.append(f'- Em **{tp}**, clientes da unidade descrevem situações como: "{ex[0]}".')
            var_row = variability_for_touchpoint(tp)
            if var_row is not None:
                network_parts.append(f"- **{tp}** na rede: **{var_row['variability_type']}**.")
        lines = [intro, "", "**Os 3 maiores problemas da sua unidade agora**", md_join(bullets) if bullets else "- Não há base suficiente para apontar os 3 maiores problemas com segurança.", "", "**Sinais dos comentários e tags**", f"As reclamações mais recorrentes são **{top_tags(complaint_tags)}**. As sugestões mais frequentes são **{top_tags(suggestion_tags)}**."]
        if evidence_parts:
            lines += ["", "**Evidências nos comentários**", md_join(evidence_parts)]
        if network_parts:
            lines += ["", "**Comparação com a rede**", md_join(network_parts)]
        lines += ["", "**Recomendação executiva**", "Monte plano de ação da unidade com dono, prazo e acompanhamento semanal para esses 3 temas. Onde a rede mostrar alta variabilidade, trate como disciplina de execução local; onde o padrão for estrutural, escale o tema."]
        return md_join(lines)

    if "caiu" in q:
        if weekly is None or len(weekly) < 2:
            return md_join([intro, "", "Ainda não há períodos suficientes para explicar queda de NPS com segurança."])
        weeks_lines = [f"- **{r['week']}**: NPS **{r['nps']}** com **{int(r['respostas'])}** respostas" for _, r in weekly.tail(4).iterrows()]
        touch_text = [f"- **{row['touchpoint']}** segue entre os touchpoints mais sensíveis para explicar piora de percepção na sua unidade." for _, row in priorities.head(3).iterrows()]
        recent_examples = find_comment_examples(comment_evidence, touchpoint=None, nps_class="Detrator", limit=2)
        if pd.notna(delta) and delta < 0:
            lines = [intro, "", f"**Queda recente identificada na unidade**\nSeu NPS caiu **{abs(round(delta, 1))}** pontos na comparação entre os dois períodos mais recentes.", "", "**Evolução recente**", md_join(weeks_lines), "", "**Hipótese principal**", f"A queda da sua unidade não deve ser lida só como oscilação estatística. Ela precisa ser investigada a partir dos touchpoints mais críticos e das reclamações mais recorrentes, principalmente **{top_tags(complaint_tags)}**.", "", "**Onde olhar primeiro**", md_join(touch_text)]
            if recent_examples:
                lines += ["", "**Evidências nos comentários**", md_join([f'- "{e}"' for e in recent_examples])]
            lines += ["", "**Comparação com a rede**", "Compare a trajetória da sua unidade com a média da rede para entender se a queda é local ou parte de um padrão mais amplo."]
            return md_join(lines)
        return md_join([intro, "", "**Leitura temporal**", "Não identifiquei uma queda recente clara no NPS da sua unidade.", "", "**Evolução recente**", md_join(weeks_lines)])

    return md_join([intro, "", "**Leitura inicial da unidade**", f"Os principais temas de reclamação são **{top_tags(complaint_tags)}**, enquanto os temas positivos mais recorrentes são **{top_tags(compliment_tags)}**.", "", "**Perguntas que o MVP já responde melhor para o gerente**", "- Quais são os maiores pontos de atenção da minha unidade?", "- O que fazer para melhorar o NPS da minha unidade?", "- Qual é o maior diferencial da minha unidade?", "- Quais são os 3 maiores problemas da minha unidade?", "- Por que o NPS da minha unidade caiu?"])


def load_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    return pd.read_csv(uploaded_file) if name.endswith(".csv") else pd.read_excel(uploaded_file)


def main():
    st.title("WeHelp Retention MVP")
    st.caption("MVP simples para leitura da planilha, cálculo de NPS, odds ratio e respostas guiadas para gerentes de unidade.")
    with st.sidebar:
        benchmark = st.number_input("Benchmark de NPS", min_value=-100.0, max_value=100.0, value=50.0, step=1.0)

    uploaded_file = st.file_uploader("Suba a planilha limpa em XLSX ou CSV", type=["xlsx", "csv"])
    if not uploaded_file:
        st.info("Suba um arquivo para começar.")
        return

    raw = load_file(uploaded_file)
    schema = detect_schema(raw)
    data = prepare_dataframe(raw, schema)
    unit_options = ["Todas as unidades"] + sorted([u for u in data["unit_clean"].dropna().unique().tolist() if str(u).strip()])
    selected_unit = st.selectbox("Selecione a unidade do gerente", options=unit_options, index=0)
    filtered = data if selected_unit == "Todas as unidades" else data[data["unit_clean"] == selected_unit]
    insights = build_insights(data, schema, benchmark, selected_unit=selected_unit)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("NPS da Unidade", insights["overall_nps"])
    c2.metric("Zona", insights["zone"])
    c3.metric("Gap vs Benchmark", insights["benchmark_gap"])
    c4.metric("Gap vs Rede" if selected_unit != "Todas as unidades" and pd.notna(insights["unit_vs_network_gap"]) else "Respostas", insights["unit_vs_network_gap"] if selected_unit != "Todas as unidades" and pd.notna(insights["unit_vs_network_gap"]) else len(filtered))

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Resumo executivo", "Segmentações", "Touchpoints", "Rede vs Unidade", "Tags e comentários", "Perguntas ao agente"])

    with tab1:
        st.markdown(answer_question("quais são os maiores pontos de atenção da minha unidade", insights))
        st.dataframe(insights["top_priorities"], use_container_width=True)

    with tab2:
        st.dataframe(insights["period"], use_container_width=True)
        st.dataframe(insights["units"], use_container_width=True)
        st.dataframe(segment_nps(filtered, "age_band"), use_container_width=True)
        st.dataframe(segment_nps(filtered, "tenure_band"), use_container_width=True)
        st.dataframe(segment_nps(filtered, "gender_clean"), use_container_width=True)
        st.dataframe(segment_nps(filtered, "plan_type_clean"), use_container_width=True)
        st.dataframe(segment_nps(filtered, "plan_name_clean"), use_container_width=True)
        st.dataframe(insights["weekly"], use_container_width=True)

    with tab3:
        st.dataframe(insights["touchpoints"], use_container_width=True)

    with tab4:
        if selected_unit == "Todas as unidades":
            st.info("Selecione uma unidade específica para ver o comparativo do gerente contra a rede.")
        else:
            st.markdown(f"**Unidade selecionada:** {selected_unit}")
            st.markdown(f"**NPS da unidade:** {insights['overall_nps']}")
            st.markdown(f"**NPS da rede:** {insights['network_nps']}")
            st.markdown(f"**Gap da unidade vs rede:** {insights['unit_vs_network_gap']}")
            st.dataframe(insights["network_variability"], use_container_width=True)

    with tab5:
        st.dataframe(insights["tags"], use_container_width=True)
        st.dataframe(insights["comment_evidence"].head(50), use_container_width=True)

    with tab6:
        q = st.text_input("Digite sua pergunta", value="Quais são os maiores pontos de atenção da minha unidade?")
        if st.button("Responder"):
            st.markdown(answer_question(q, insights))


if __name__ == "__main__":
    main()
