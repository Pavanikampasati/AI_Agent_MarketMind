import os
import json
from typing import Dict, Any, List, Optional
from groq import Groq
from models.schemas import FinalReport, ReportSection, Finding, Source, SourceType, QuestionType
from agent.planner import create_groq_completion

REPORT_SYSTEM_PROMPT = """You are a world-class business research analyst and management consultant.
Synthesize the collected research evidence and findings into a custom, dynamic Business Intelligence Report tailored specifically to the user's question.

CRITICAL INSTRUCTIONS:
1. NO HARDCODED TOPICS: Generate sections and insights strictly based on the actual question and gathered findings.
2. DYNAMIC SECTIONS: Construct sections corresponding to the target section titles requested.
3. CITATION GROUNDING: Every claim, data point, or insight MUST be grounded in the provided evidence.
4. SECTION CONTENT FORMATS:
   - Use "text" for analytical markdown narrative paragraphs.
   - Use "table" for structured data comparisons (list of objects with consistent keys, e.g. [{"Competitor": "X", "Pricing": "$Y", "Feature": "Z"}]).
   - Use "key_metrics" for numerical KPIs or financial metrics (dict of metric_name -> metric_value).
   - Use "bullet_list" for bulleted strategic insights or risk factors (list of strings).
5. DYNAMIC CHART DATA:
   If concrete, factual numerical data is present in the evidence (e.g. competitor pricing comparisons, revenue per product, market shares, pricing tiers, numerical KPIs), include a "chart_data" object:
   "chart_data": {
       "chart_type": "bar",
       "title": "Descriptive Chart Title",
       "x_label": "Label for Categories/Items",
       "y_label": "Label for Numerical Values",
       "data": [
           {"label": "Item A", "value": 150.0},
           {"label": "Item B", "value": 95.5}
       ],
       "source": "Source Title or Dataset"
   }
   CRITICAL: NEVER invent or fabricate numerical values. Only extract real numbers present in the evidence. If no concrete numerical data exists in the evidence, set "chart_data": null.
6. GAPS & LIMITATIONS: If research could not obtain specific details after reasonable execution, explicitly list them under "gaps_and_limitations" rather than guessing.

Return ONLY a valid JSON object matching:
{
  "title": "Report Title customized for the question",
  "executive_summary": "Comprehensive high-level synthesis directly answering the user's core question...",
  "sections": [
    {
      "title": "Exact Title of Section",
      "section_type": "text | table | key_metrics | bullet_list",
      "content": "Markdown text OR list of dicts for table OR dict for metrics OR list of strings",
      "sources": ["Title of Source 1", "URL or Document Name 2"]
    }
  ],
  "key_takeaways": [
    "Actionable takeaway 1",
    "Actionable takeaway 2"
  ],
  "chart_data": null,
  "gaps_and_limitations": [
    "Note on any genuinely unavailable information (omit or empty list if none)"
  ]
}
"""

def _fallback_extract_chart(sections: List[ReportSection], findings: List[Finding]) -> Optional[Dict[str, Any]]:
    # Check table sections
    for sec in sections:
        if sec.section_type == "table" and isinstance(sec.content, list) and len(sec.content) >= 2:
            first_row = sec.content[0]
            if isinstance(first_row, dict):
                cat_col = None
                num_col = None
                for k in first_row.keys():
                    is_num = True
                    for r in sec.content:
                        val = r.get(k)
                        if isinstance(val, (int, float)):
                            pass
                        elif isinstance(val, str):
                            cleaned = val.replace("$", "").replace("₹", "").replace(",", "").replace("%", "").strip()
                            try:
                                float(cleaned)
                            except ValueError:
                                is_num = False
                        else:
                            is_num = False
                    if is_num and num_col is None:
                        num_col = k
                    elif not is_num and cat_col is None:
                        cat_col = k

                if cat_col and num_col:
                    data_items = []
                    for r in sec.content:
                        lbl = str(r.get(cat_col, "")).strip()
                        val_raw = r.get(num_col)
                        val_num = None
                        if isinstance(val_raw, (int, float)):
                            val_num = float(val_raw)
                        elif isinstance(val_raw, str):
                            cleaned = val_raw.replace("$", "").replace("₹", "").replace(",", "").replace("%", "").strip()
                            try:
                                val_num = float(cleaned)
                            except ValueError:
                                pass
                        if lbl and val_num is not None:
                            data_items.append({"label": lbl, "value": val_num})

                    if len(data_items) >= 2:
                        return {
                            "chart_type": "bar",
                            "title": f"{sec.title} ({num_col})",
                            "x_label": cat_col,
                            "y_label": num_col,
                            "data": data_items,
                            "source": sec.sources[0] if sec.sources else "Report Table"
                        }

    # Check key_metrics sections
    for sec in sections:
        if sec.section_type == "key_metrics" and isinstance(sec.content, dict) and len(sec.content) >= 2:
            data_items = []
            for k, v in sec.content.items():
                val_num = None
                if isinstance(v, (int, float)):
                    val_num = float(v)
                elif isinstance(v, str):
                    cleaned = v.replace("$", "").replace("₹", "").replace(",", "").replace("%", "").strip()
                    try:
                        val_num = float(cleaned)
                    except ValueError:
                        pass
                if val_num is not None:
                    data_items.append({"label": str(k), "value": val_num})
            if len(data_items) >= 2:
                return {
                    "chart_type": "bar",
                    "title": sec.title,
                    "x_label": "Metric",
                    "y_label": "Value",
                    "data": data_items,
                    "source": sec.sources[0] if sec.sources else "Key Metrics"
                }

    return None

def generate_report(
    research_question: str,
    question_type: QuestionType,
    user_intent: str,
    recommended_sections: List[str],
    findings: List[Finding],
    sources: List[Source],
    groq_api_key: Optional[str] = None
) -> FinalReport:
    """
    Synthesizes findings into a dynamic FinalReport with dynamic sections matching the question type.
    """
    api_key = groq_api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is missing.")

    client = Groq(api_key=api_key)

    # Format findings with source attributions
    formatted_findings = []
    for idx, f in enumerate(findings):
        src_info = f"{f.source.title} ({f.source.source_type.value})"
        if f.source.url:
            src_info += f" [{f.source.url}]"
        formatted_findings.append(
            f"Finding #{idx+1} [Task {f.task_id}] (Category: {f.category}):\n"
            f"Insight: {f.finding}\n"
            f"Source: {src_info}\n"
            f"Context Quote: \"{f.source.supporting_context[:300]}\"\n"
        )

    findings_text = "\n---\n".join(formatted_findings) if formatted_findings else "No specific findings collected."

    user_prompt = f"""User Question: {research_question}
Question Type: {question_type.value}
User Intent: {user_intent}
Target Report Sections: {json.dumps(recommended_sections)}

EVIDENCE & FINDINGS COLLECTED:
{findings_text}

Generate the dynamic report JSON now.
"""

    try:
        content = create_groq_completion(client, REPORT_SYSTEM_PROMPT, user_prompt, json_mode=True, temperature=0.2)
        data = json.loads(content)

        sections = []
        for sec_data in data.get("sections", []):
            title = sec_data.get("title", "Analysis Section")
            stype = sec_data.get("section_type", "text")
            raw_c = sec_data.get("content", "")
            sec_sources = sec_data.get("sources", [])

            sections.append(ReportSection(
                title=title,
                section_type=stype,
                content=raw_c,
                sources=sec_sources
            ))

        # Fallback section if LLM returned no sections
        if not sections:
            sections.append(ReportSection(
                title="Detailed Research Findings",
                section_type="bullet_list",
                content=[f.finding for f in findings] if findings else ["Analysis completed."],
                sources=[s.title for s in sources]
            ))

        chart_data = data.get("chart_data")
        if not isinstance(chart_data, dict) or not chart_data.get("data"):
            chart_data = _fallback_extract_chart(sections, findings)

        return FinalReport(
            title=data.get("title", f"Research Report: {research_question}"),
            executive_summary=data.get("executive_summary", f"Executive synthesis for {research_question}"),
            question_type=question_type,
            sections=sections,
            key_takeaways=data.get("key_takeaways", []),
            chart_data=chart_data,
            gaps_and_limitations=data.get("gaps_and_limitations") if data.get("gaps_and_limitations") else None,
            sources=sources
        )
    except Exception as e:
        print(f"[Report Generator Warning] Fallback report generated due to: {e}")
        # Build clean dynamic fallback report
        sec_content = []
        for f in findings:
            sec_content.append(f"**[{f.category}]** {f.finding} *(Source: {f.source.title})*")

        sections = [
            ReportSection(
                title="Key Evidence & Findings",
                section_type="bullet_list",
                content=sec_content if sec_content else ["Research query executed successfully."],
                sources=[s.title for s in sources]
            )
        ]
        chart_data = _fallback_extract_chart(sections, findings)

        return FinalReport(
            title=f"Business Intelligence Report: {research_question}",
            executive_summary=f"Analysis conducted for: '{research_question}'. Total of {len(findings)} findings compiled across {len(sources)} sources.",
            question_type=question_type,
            sections=sections,
            key_takeaways=[
                f"Completed research for question type: {question_type.value}",
                f"Utilized {len(sources)} evidence sources for analysis"
            ],
            chart_data=chart_data,
            gaps_and_limitations=None,
            sources=sources
        )
