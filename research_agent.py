import os
import json
from typing import Callable, Optional, Dict, Any, List
from groq import Groq

from models.schemas import (
    ResearchState, ResearchPlan, ResearchTask, TaskStatus, Finding, Source, SourceType,
    FinalReport, QuestionType
)
from agent.state import StateManager
from agent.planner import generate_plan, create_groq_completion
from tools.web_search import search_web
from tools.web_retrieval import retrieve_webpage
from tools.rag_tool import search_knowledge_base
from tools.pandas_tool import analyze_tabular_file
from tools.calculator_tool import execute_calculation
from rag.retriever import RAGRetriever
from reports.report_generator import generate_report

EVALUATOR_SYSTEM_PROMPT = """You are an expert research analyst evaluating raw tool execution results for a specific business task.
Your goal is to extract 2 to 4 factual, highly relevant business findings from the raw tool output.

For EACH finding:
- Extract the core factual insight, data point, or calculated metric.
- Categorize the insight (e.g., "Market Dynamics", "Competitor Pricing", "Quantitative Metric", "Document Context", "Risk Factor").
- Identify the exact source title and URL or document filename.
- Classify source type as one of: 'web', 'uploaded_document', 'web_retrieval', 'data_analysis', or 'calculator'.
- Provide a direct context snippet or data snippet supporting the finding.

Return ONLY a valid JSON object matching:
{
  "findings": [
    {
      "finding": "Text describing the key factual finding or data metric",
      "category": "Market Dynamics OR Competitor Pricing OR Quantitative Metric etc.",
      "source_title": "Exact title of website, article, or file",
      "source_url": "URL if web, or file path/name if uploaded document/spreadsheet",
      "source_type": "web OR uploaded_document OR web_retrieval OR data_analysis OR calculator",
      "supporting_context": "Exact quote or snippet supporting the finding"
    }
  ]
}

CRITICAL: Do NOT return empty source_title fields. If the tool output provides a URL or document title, USE IT.
"""

GAP_ANALYZER_SYSTEM_PROMPT = """You are a research quality auditor checking if the collected research findings adequately answer the user's question.
If key information is missing and CAN be found via web research, propose ONE alternative follow-up search query.

User Question: {question}
Collected Findings: {findings_summary}

If findings are already sufficient to provide a clear, evidence-based answer, return:
{"needs_re_research": false, "followup_task_description": "", "alternative_query": ""}

If critical information is missing, return:
{"needs_re_research": true, "followup_task_description": "Investigate missing details regarding...", "alternative_query": "specific search query"}
"""

class ResearchAgent:
    """
    GENERAL-PURPOSE AGENT implementing the 10-Step Dynamic Execution Loop:
    1. Understand intent
    2. Classify question type
    3. Strategic planning
    4. Task decomposition
    5. Dynamic tool selection
    6. Tool execution
    7. Evidence collection & validation
    8. Evidence analysis
    9. Gap check & re-research
    10. Dynamic report generation
    """
    def __init__(self, groq_api_key: Optional[str] = None, retriever: Optional[RAGRetriever] = None, tabular_files: Optional[List[str]] = None):
        self.api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is missing in server environment.")
        self.client = Groq(api_key=self.api_key)
        self.retriever = retriever
        self.tabular_files = tabular_files or []

    def run_research(self, research_question: str, step_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None) -> ResearchState:
        """
        Executes end-to-end research workflow dynamically.
        """
        state_mgr = StateManager(research_question)
        has_docs = self.retriever is not None
        has_tabular = len(self.tabular_files) > 0

        # STEP 1, 2, 3, 4: UNDERSTAND INTENT, CLASSIFY, PLAN, & DECOMPOSE TASKS
        if step_callback:
            step_callback("PLANNING_START", {"question": research_question})

        plan = generate_plan(
            research_question,
            has_uploaded_docs=has_docs,
            has_tabular_files=has_tabular,
            groq_api_key=self.api_key
        )
        state_mgr.set_plan(plan)

        if step_callback:
            step_callback("PLANNING_COMPLETE", {
                "question_type": plan.question_type.value,
                "intent": plan.user_intent,
                "objective": plan.research_objective,
                "tasks": plan.tasks
            })

        # STEP 5, 6, 7, 8: TOOL SELECTION, EXECUTION, EVIDENCE COLLECTION & ANALYSIS LOOP
        max_total_tasks = 8
        task_count = 0

        while task_count < max_total_tasks:
            current_task = state_mgr.start_next_task()
            if not current_task:
                # STEP 9: DETERMINE IF ADDITIONAL RESEARCH IS REQUIRED (GAP ANALYSIS)
                if state_mgr.state.re_research_attempts < 1 and len(state_mgr.state.findings) > 0:
                    gap_check = self._check_research_gaps(research_question, state_mgr.state.findings)
                    if gap_check.get("needs_re_research") and gap_check.get("alternative_query"):
                        re_task = ResearchTask(
                            task_id=f"T_RE{state_mgr.state.re_research_attempts+1}",
                            description=gap_check.get("followup_task_description", f"Search additional evidence for {research_question}"),
                            purpose="Fill identified information gap from primary research phase",
                            status=TaskStatus.PENDING,
                            assigned_tool="web_search"
                        )
                        state_mgr.add_re_research_task(re_task)
                        if step_callback:
                            step_callback("RE_RESEARCH_TRIGGERED", {
                                "gap_reason": gap_check.get("followup_task_description"),
                                "query": gap_check.get("alternative_query")
                            })
                        continue
                break

            task_count += 1
            task_id = current_task.task_id
            tool_name = current_task.assigned_tool

            if step_callback:
                step_callback("TASK_START", {
                    "task": current_task,
                    "state": state_mgr.get_state_summary()
                })

            try:
                # EXECUTE SELECTED TOOL
                tool_output, query_used = self._execute_tool(current_task, research_question)
                state_mgr.add_tool_result(tool_name, task_id, query_used, tool_output)

                if step_callback:
                    step_callback("TOOL_EXECUTED", {
                        "task_id": task_id,
                        "tool": tool_name,
                        "query": query_used,
                        "raw_output_snippet": str(tool_output)[:300]
                    })

                # EVALUATE TOOL OUTPUT & EXTRACT STRUCTURED EVIDENCE FINDINGS
                extracted_findings = self._evaluate_tool_output(current_task, tool_output, tool_name)

                # COMPLETE TASK & UPDATE STATE
                summary_str = f"Extracted {len(extracted_findings)} evidence findings via {tool_name}."
                state_mgr.complete_current_task(task_id, summary_str, extracted_findings)

                if step_callback:
                    step_callback("TASK_COMPLETE", {
                        "task_id": task_id,
                        "findings": extracted_findings,
                        "state": state_mgr.get_state_summary()
                    })

            except Exception as e:
                error_msg = str(e)
                state_mgr.fail_current_task(task_id, error_msg)
                if step_callback:
                    step_callback("TASK_FAILED", {"task_id": task_id, "error": error_msg})

        # STEP 10: DYNAMIC REPORT GENERATION tailored to user's question
        if step_callback:
            step_callback("REPORT_GENERATION_START", {
                "question_type": state_mgr.state.question_type.value,
                "findings_count": len(state_mgr.state.findings),
                "sections": plan.recommended_report_sections
            })

        final_report = generate_report(
            research_question=research_question,
            question_type=state_mgr.state.question_type,
            user_intent=state_mgr.state.user_intent,
            recommended_sections=plan.recommended_report_sections,
            findings=state_mgr.state.findings,
            sources=state_mgr.state.sources,
            groq_api_key=self.api_key
        )

        state_mgr.set_final_report(final_report)

        if step_callback:
            step_callback("RESEARCH_COMPLETE", {
                "final_report": final_report,
                "state": state_mgr.state
            })

        return state_mgr.state

    def _execute_tool(self, task: ResearchTask, research_question: str) -> tuple[Any, str]:
        """
        Dynamic Tool Selector and Executor.
        Returns tuple of (tool_output, query_or_target_used).
        """
        tool_name = task.assigned_tool.lower()

        if "data_analysis" in tool_name or "excel" in tool_name or "pandas" in tool_name or "csv" in tool_name:
            if self.tabular_files:
                target_file = self.tabular_files[0]
                output = analyze_tabular_file(target_file, f"{task.description} {research_question}")
                return output, target_file
            else:
                query = f"{task.description} {research_question}"
                output = search_web(query, max_results=5)
                return output, query

        elif "rag" in tool_name or "knowledge" in tool_name or "document" in tool_name:
            if self.retriever:
                query = f"{task.description} {research_question}"
                output = search_knowledge_base(query, retriever=self.retriever, top_k=5)
                return output, query
            else:
                query = f"{task.description} {research_question}"
                output = search_web(query, max_results=5)
                return output, query

        elif "calculator" in tool_name or "calc" in tool_name or "math" in tool_name:
            output = execute_calculation(task.description)
            return output, task.description

        elif "retrieval" in tool_name or "scrape" in tool_name:
            search_res = search_web(f"{task.description}", max_results=3)
            target_url = search_res[0].get("url") if search_res and search_res[0].get("url") else "https://en.wikipedia.org"
            output = retrieve_webpage(target_url)
            return output, target_url

        else: # Default: web_search
            query = f"{task.description} {research_question}"
            output = search_web(query, max_results=5)
            return output, query

    def _evaluate_tool_output(self, task: ResearchTask, raw_output: Any, tool_name: str) -> List[Finding]:
        """
        Evaluates tool execution output and extracts structured Finding objects.
        """
        prompt = f"""Task Description: {task.description}
Tool Assigned: {tool_name}
Raw Tool Execution Output:
{json.dumps(raw_output, indent=2, default=str)[:4000]}

Extract 2 to 4 concise factual findings with precise source attributions.
"""

        try:
            content = create_groq_completion(self.client, EVALUATOR_SYSTEM_PROMPT, prompt, json_mode=True, temperature=0.1)
            data = json.loads(content)

            findings_list = []
            for item in data.get("findings", []):
                st_str = item.get("source_type", "web").lower()
                if "data" in st_str or tool_name == "data_analysis":
                    st = SourceType.DATA_ANALYSIS
                elif "doc" in st_str or tool_name == "rag_search":
                    st = SourceType.UPLOADED_DOCUMENT
                elif "retrieval" in st_str or tool_name == "webpage_retrieval":
                    st = SourceType.WEB_RETRIEVAL
                elif "calc" in st_str or tool_name == "calculator":
                    st = SourceType.CALCULATOR
                else:
                    st = SourceType.WEB

                title = item.get("source_title") or "Research Source"
                url = item.get("source_url") or None

                # Extract URL from raw output if missing
                if not url and isinstance(raw_output, list) and raw_output and isinstance(raw_output[0], dict):
                    url = raw_output[0].get("url")
                    if not title or title == "Research Source":
                        title = raw_output[0].get("title", title)

                source = Source(
                    title=title,
                    url=url,
                    source_type=st,
                    supporting_context=item.get("supporting_context", "Extracted from research execution.")
                )
                findings_list.append(Finding(
                    task_id=task.task_id,
                    finding=item.get("finding", "Factual insight extracted."),
                    category=item.get("category", "General Insight"),
                    source=source
                ))

            return findings_list if findings_list else [self._default_finding(task, tool_name, raw_output)]
        except Exception as e:
            print(f"[Evaluator Warning] {e}")
            return [self._default_finding(task, tool_name, raw_output)]

    def _default_finding(self, task: ResearchTask, tool_name: str, raw_output: Any) -> Finding:
        title = "Business Intelligence Source"
        url = None
        if isinstance(raw_output, list) and raw_output and isinstance(raw_output[0], dict):
            title = raw_output[0].get("title", title)
            url = raw_output[0].get("url")

        st = SourceType.DATA_ANALYSIS if tool_name == "data_analysis" else (
            SourceType.UPLOADED_DOCUMENT if tool_name == "rag_search" else (
                SourceType.CALCULATOR if tool_name == "calculator" else SourceType.WEB
            )
        )
        return Finding(
            task_id=task.task_id,
            finding=f"Synthesized evidence for: {task.description}",
            category="Research Evidence",
            source=Source(
                title=title,
                url=url,
                source_type=st,
                supporting_context="Generated during dynamic task execution loop."
            )
        )

    def _check_research_gaps(self, question: str, findings: List[Finding]) -> Dict[str, Any]:
        """
        Evaluates whether collected findings leave gaps for the question.
        """
        findings_summary = "\n".join([f"- [{f.category}] {f.finding}" for f in findings[:6]])
        prompt = f"User Question: {question}\nFindings Summary:\n{findings_summary}"
        try:
            content = create_groq_completion(self.client, GAP_ANALYZER_SYSTEM_PROMPT.format(question=question, findings_summary=findings_summary), prompt, json_mode=True, temperature=0.1)
            return json.loads(content)
        except Exception:
            return {"needs_re_research": False}
