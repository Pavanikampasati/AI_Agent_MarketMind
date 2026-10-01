import os
import tempfile
import json
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load server environment variables
load_dotenv()

from models.schemas import ResearchState, QuestionType, SourceType
from rag.vector_store import VectorStoreManager
from rag.retriever import RAGRetriever
from agent.research_agent import ResearchAgent

# Streamlit Page Config
st.set_page_config(
    page_title="AI Business Research & Market Intelligence Agent",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1E293B 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    
    /* Card Container */
    .report-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .report-card:hover {
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    
    /* Badges */
    .badge-web {
        background-color: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
        padding: 4px 10px;
        border-radius: 14px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .badge-doc {
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FDE68A;
        padding: 4px 10px;
        border-radius: 14px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .badge-data {
        background-color: #F0FDF4;
        color: #15803D;
        border: 1px solid #BBF7D0;
        padding: 4px 10px;
        border-radius: 14px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .badge-calc {
        background-color: #F3E8FF;
        color: #7E22CE;
        border: 1px solid #E9D5FF;
        padding: 4px 10px;
        border-radius: 14px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .qtype-badge {
        display: inline-block;
        background-color: #0F172A;
        color: #F8FAFC;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 12px;
    }
    
    /* Custom Preset Buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStoreManager()
if "retriever" not in st.session_state:
    st.session_state.retriever = RAGRetriever(st.session_state.vector_store)
if "uploaded_doc_names" not in st.session_state:
    st.session_state.uploaded_doc_names = []
if "tabular_file_paths" not in st.session_state:
    st.session_state.tabular_file_paths = []
if "research_state" not in st.session_state:
    st.session_state.research_state = None
if "research_question" not in st.session_state:
    st.session_state["research_question"] = ""

with st.sidebar:

    st.markdown("# MARKETMIND")
    st.caption("AI Business Intelligence")

    st.title("💡 Research Insights")

    state = st.session_state.research_state

    if state:
        q_type_str = state.question_type.value.replace("_", " ").title() if hasattr(state.question_type, 'value') else str(state.question_type).replace("_", " ").title()
        unique_files_count = len(st.session_state.uploaded_doc_names)
        total_sources = len(state.sources)
        total_findings = len(state.findings)

        tool_names = set()
        for t in state.tasks:
            if t.assigned_tool:
                tool_names.add(t.assigned_tool)

        tool_mapping = {
            "web_search": "Web Search",
            "webpage_retrieval": "Web Page Retrieval",
            "rag_search": "RAG Search",
            "data_analysis": "Data Analysis",
            "calculator": "Calculator"
        }

        tools_formatted = ", ".join([tool_mapping.get(t, t) for t in tool_names]) if tool_names else "None"
        status_str = "Completed" if state.final_report else "In Progress"

        st.markdown(f"**📌 Question Type**  \n{q_type_str}")
        st.markdown(f"**📄 Files Analyzed**  \n{unique_files_count}")
        st.markdown(f"**🔎 Sources Retrieved**  \n{total_sources}")
        st.markdown(f"**🧠 Evidence Findings**  \n{total_findings}")
        st.markdown(f"**🔧 Tools Used**  \n{tools_formatted}")
        st.markdown(f"**📊 Research Status**  \n{status_str}")

    else:
        unique_files_count = len(st.session_state.uploaded_doc_names)
        st.markdown(f"**📌 Question Type**  \nIdle")
        st.markdown(f"**📄 Files Analyzed**  \n{unique_files_count}")
        st.markdown(f"**🔎 Sources Retrieved**  \n0")
        st.markdown(f"**🧠 Evidence Findings**  \n0")
        st.markdown(f"**🔧 Tools Used**  \nNone")
        st.markdown(f"**📊 Research Status**  \nIdle")
    st.divider()

    st.subheader("📊 Research Visualization")

    chart_rendered = False
    if state and state.final_report and hasattr(state.final_report, 'chart_data') and state.final_report.chart_data:
        cdata = state.final_report.chart_data
        if isinstance(cdata, dict) and "data" in cdata and isinstance(cdata["data"], list) and len(cdata["data"]) > 0:
            try:
                chart_title = cdata.get("title", "Research Data Visualization")
                st.caption(f"**{chart_title}**")

                df_chart = pd.DataFrame(cdata["data"])
                if "label" in df_chart.columns and "value" in df_chart.columns:
                    df_chart["value"] = pd.to_numeric(df_chart["value"], errors="coerce")
                    df_chart = df_chart.dropna(subset=["value"])

                    if not df_chart.empty:
                        df_chart_indexed = df_chart.set_index("label")[["value"]]
                        st.bar_chart(df_chart_indexed)
                        if cdata.get("source"):
                            st.caption(f"*Source: {cdata.get('source')}*")
                        chart_rendered = True
            except Exception as e:
                chart_rendered = False

    if not chart_rendered:
        st.info("📊 No suitable numerical data available for visualization.")

    st.divider()

    st.subheader("💡 Sample Question Presets")
    st.caption("Click any preset to load a question type:")

    presets = [
        ("📊 Market Research", "Analyze the Indian electric vehicle market and identify major competitors, pricing, market trends, opportunities, and risks."),
        ("⚔️ Competitor Research", "Identify the major competitors of Netflix in India and compare their services, pricing, target customers, and key differentiators."),
        ("💰 Pricing Research", "Compare the pricing and major features of AWS, Microsoft Azure, and Google Cloud for a small startup."),
        ("🚀 Trend Research", "What are the major trends currently shaping the sustainable fashion industry? Explain the key drivers and business opportunities."),
        ("⚠️ Business Analysis", "What are the major opportunities and risks for launching a food delivery startup in India?"),
        ("📄 Document Analysis", "Analyze the uploaded document and identify the most important business findings, opportunities, risks, and key insights."),
        ("📈 Spreadsheet Analysis", "Analyze the uploaded dataset and identify the highest-revenue products and important patterns in the data."),
        ("🔄 Mixed Research", "Analyze the uploaded data and compare the findings with current market trends and relevant external research.")
    ]

    for label, query in presets:
        if st.button(label, use_container_width=True, key=f"btn_{label}"):
            st.session_state["research_question"] = query
            st.rerun()

    st.divider()
    if st.button("🧹 Reset All State", type="secondary", use_container_width=True):
        st.session_state.research_state = None
        st.session_state.uploaded_doc_names = []
        st.session_state.tabular_file_paths = []
        st.session_state["research_question"] = ""
        st.success("Session state reset.")
        st.rerun()

# Header Section
st.markdown('<div class="main-title">AI Business Research & Market Intelligence Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">General-Purpose Autonomous Business Analyst — Submit any business query, dataset, or research document for evidence-based intelligence.</div>', unsafe_allow_html=True)

# Upload Section (Optional)
with st.expander("📁 Upload Business Documents or Datasets (Optional)", expanded=len(st.session_state.uploaded_doc_names) > 0):
    uploaded_files = st.file_uploader(
        "Upload Text Documents (PDF, DOCX, TXT) or Tabular Datasets (CSV, XLSX, XLS):",
        type=["pdf", "docx", "txt", "xlsx", "xls", "csv"],
        accept_multiple_files=True,
        help="Text files are indexed into RAG Vector Store. Spreadsheets are processed via Pandas."
    )

    if uploaded_files:
        tabular_files_list = []
        for file in uploaded_files:
            filename = file.name
            ext = os.path.splitext(filename)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(file.getvalue())
                tmp_path = tmp.name

            if ext in [".csv", ".xlsx", ".xls"]:
                tabular_files_list.append(tmp_path)
                if filename not in st.session_state.uploaded_doc_names:
                    st.session_state.uploaded_doc_names.append(filename)
            else:
                if filename not in st.session_state.uploaded_doc_names:
                    st.session_state.retriever.process_and_index_file(tmp_path)
                    st.session_state.uploaded_doc_names.append(filename)

        st.session_state.tabular_file_paths = tabular_files_list
        st.success(f"Loaded {len(uploaded_files)} file(s): {', '.join([f.name for f in uploaded_files])}")

# Query Input Box
st.subheader("💬 Enter Business Research Question")
user_query = st.text_area(
    "Research Question:",
    height=100,
    placeholder="Ask any market, competitor, pricing, trend, risk, document, or spreadsheet analysis question...",
    key="research_question"
)

col_btn, col_blank = st.columns([1.5, 4])
with col_btn:
    analyze_btn = st.button("🔍 Execute Research", type="primary", use_container_width=True)

# Execution Workflow
if analyze_btn:
    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        st.error("Cannot proceed: GROQ_API_KEY is not configured in server environment.")
    elif not user_query.strip():
        st.warning("Please enter a research question.")
    else:
        # Reset previous research state on new execution
        st.session_state.research_state = None
        st.session_state.input_question = user_query.strip()

        status_box = st.status("🤖 Agent is executing 10-step research workflow...", expanded=True)

        def internal_callback(stage, payload):
            if stage == "PLANNING_START":
                status_box.write("1️⃣ & 2️⃣ Understanding intent & classifying question type...")
            elif stage == "PLANNING_COMPLETE":
                qtype = payload.get("question_type", "")
                tasks = payload.get("tasks", [])
                status_box.write(f"3️⃣ & 4️⃣ Created Strategic Plan (Type: `{qtype}`). Decomposed into {len(tasks)} tasks.")
            elif stage == "TASK_START":
                task = payload.get("task")
                status_box.write(f"5️⃣ & 6️⃣ Executing Task **[{task.task_id}]**: {task.description} *(Tool: `{task.assigned_tool}`)*...")
            elif stage == "TOOL_EXECUTED":
                tool = payload.get("tool")
                status_box.write(f"7️⃣ & 8️⃣ Collected & validated evidence output from `{tool}`.")
            elif stage == "RE_RESEARCH_TRIGGERED":
                reason = payload.get("gap_reason")
                status_box.write(f"9️⃣ Gap Analysis Triggered Follow-up Research: *{reason}*")
            elif stage == "REPORT_GENERATION_START":
                status_box.write("🔟 Synthesizing dynamic report tailored to question...")
            elif stage == "RESEARCH_COMPLETE":
                status_box.update(label="✅ Research Complete!", state="complete", expanded=False)

        try:
            agent = ResearchAgent(
                groq_api_key=groq_key,
                retriever=st.session_state.retriever if st.session_state.uploaded_doc_names else None,
                tabular_files=st.session_state.tabular_file_paths
            )

            final_state = agent.run_research(user_query.strip(), step_callback=internal_callback)
            st.session_state.research_state = final_state
            st.rerun()

        except Exception as e:
            status_box.update(label="❌ Analysis Failed", state="error", expanded=True)
            st.error(f"Error during execution: {str(e)}")

# Display Dynamic Report Output
if st.session_state.research_state and st.session_state.research_state.final_report:
    state: ResearchState = st.session_state.research_state
    report = state.final_report

    st.divider()

    # Title & Question Type Badge
    st.markdown(f'<div class="qtype-badge">QUESTION TYPE: {report.question_type.value.upper().replace("_", " ")}</div>', unsafe_allow_html=True)
    st.header(f"📊 {report.title}")

    # Executive Summary Card
    st.markdown("### 💡 Executive Summary")
    st.info(report.executive_summary)

    # Dynamic Report Sections
    if report.sections:
        for sec in report.sections:
            st.markdown(f"### {sec.title}")

            if sec.section_type == "table" and isinstance(sec.content, list) and len(sec.content) > 0 and isinstance(sec.content[0], dict):
                df_sec = pd.DataFrame(sec.content)
                st.dataframe(df_sec, use_container_width=True)

            elif sec.section_type == "key_metrics" and isinstance(sec.content, dict):
                cols = st.columns(min(len(sec.content), 4))
                for idx, (m_key, m_val) in enumerate(sec.content.items()):
                    with cols[idx % len(cols)]:
                        st.metric(label=str(m_key), value=str(m_val))

            elif sec.section_type == "bullet_list" and isinstance(sec.content, list):
                for item in sec.content:
                    st.write(f"• {item}")

            else:
                # Text or markdown narrative
                if isinstance(sec.content, str):
                    st.write(sec.content)
                elif isinstance(sec.content, list):
                    for item in sec.content:
                        st.write(f"• {item}")
                else:
                    st.write(str(sec.content))

            if sec.sources:
                st.caption(f"**Sources cited:** {', '.join(sec.sources)}")
            st.markdown("<br/>", unsafe_allow_html=True)

    # Key Takeaways
    if report.key_takeaways:
        st.markdown("### 🎯 Key Takeaways & Actionable Insights")
        for takeaway in report.key_takeaways:
            st.success(f"👉 {takeaway}")

    # Gaps & Limitations (Failure handling transparency)
    if report.gaps_and_limitations:
        st.warning("### ⚠️ Research Gaps & Limitations")
        st.write("The following details could not be completely verified from available evidence:")
        for gap in report.gaps_and_limitations:
            st.write(f"- {gap}")

    # Evidence Findings Explorer
    st.divider()
    st.markdown("### 📌 Collected Evidence & Source Attributions")
    if state.findings:
        for f in state.findings:
            st_val = f.source.source_type.value if hasattr(f.source.source_type, 'value') else str(f.source.source_type)
            badge_class = 'badge-data' if st_val == 'data_analysis' else ('badge-doc' if st_val == 'uploaded_document' else ('badge-calc' if st_val == 'calculator' else 'badge-web'))

            st.markdown(f"""
            <div class="report-card">
                <b>[{f.task_id}] ({f.category})</b>: {f.finding}<br/>
                <div style="margin-top: 8px;">
                    <span class="{badge_class}">{st_val.upper()}</span>
                    <small style="color: #64748B; margin-left: 8px;"><b>Source:</b> {f.source.title} {f'({f.source.url})' if f.source.url else ''}</small>
                </div>
                <div style="font-size: 0.85rem; color: #475569; margin-top: 6px; font-style: italic;">
                    "{f.source.supporting_context[:250]}..."
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Export & Download Options
    st.divider()
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        report_md = f"# {report.title}\n\n## Executive Summary\n{report.executive_summary}\n\n"
        for sec in report.sections:
            report_md += f"## {sec.title}\n{sec.content}\n\n"
        report_md += "## Key Takeaways\n" + "\n".join([f"- {t}" for t in report.key_takeaways])

        st.download_button(
            label="📥 Download Report as Markdown",
            data=report_md,
            file_name="business_research_report.md",
            mime="text/markdown",
            use_container_width=True
        )

    with col_dl2:
        report_json = json.dumps(state.model_dump(), default=str, indent=2)
        st.download_button(
            label="📥 Download Full Research Trajectory JSON",
            data=report_json,
            file_name="research_trajectory.json",
            mime="application/json",
            use_container_width=True
        )

    # Internal Process Inspector
    with st.expander("🔍 View Internal Agentic Research Trajectory (Step-by-Step Inspector)"):
        st.write(f"**Research Question:** {state.research_question}")
        st.write(f"**Determined Question Type:** {state.question_type.value}")
        st.write(f"**User Intent:** {state.user_intent}")
        st.write(f"**Re-research Attempts:** {state.re_research_attempts}")
        st.write(f"**Total Tasks Executed:** {len(state.tasks)}")

        for t in state.tasks:
            st.caption(f"- **[{t.task_id}]** {t.description} *(Tool: `{t.assigned_tool}` | Status: `{t.status.value}`)*")

        if state.tool_results:
            st.markdown("#### Raw Tool Execution Logs:")
            st.json(state.tool_results)
