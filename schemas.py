from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum

class QuestionType(str, Enum):
    MARKET_RESEARCH = "market_research"
    COMPETITOR_RESEARCH = "competitor_research"
    PRICING_RESEARCH = "pricing_research"
    TREND_RESEARCH = "trend_research"
    BUSINESS_ANALYSIS = "business_analysis"
    DOCUMENT_ANALYSIS = "document_analysis"
    SPREADSHEET_ANALYSIS = "spreadsheet_analysis"
    MIXED_RESEARCH = "mixed_research"
    CALCULATION_ANALYSIS = "calculation_analysis"
    GENERAL_BUSINESS_QUESTION = "general_business_question"

class SourceType(str, Enum):
    WEB = "web"
    UPLOADED_DOCUMENT = "uploaded_document"
    WEB_RETRIEVAL = "web_retrieval"
    DATA_ANALYSIS = "data_analysis"
    CALCULATOR = "calculator"

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class WebSearchResult(BaseModel):
    title: str = Field(description="Title of the web page or search result")
    url: str = Field(description="URL of the web source")
    snippet: str = Field(description="Text snippet extracted from search result")
    content: Optional[str] = Field(default=None, description="Full or parsed text content if retrieved")
    source_type: SourceType = Field(default=SourceType.WEB, description="Source classification")

class Source(BaseModel):
    title: str = Field(description="Title or identifier of the source")
    url: Optional[str] = Field(default=None, description="URL of the web source if available")
    source_type: SourceType = Field(description="Classification of the source")
    supporting_context: str = Field(description="Direct snippet or quote from the source supporting the finding")

class Finding(BaseModel):
    task_id: str = Field(description="ID of the research task this finding satisfies")
    finding: str = Field(description="Core text insight or factual finding")
    category: str = Field(default="General Insight", description="Category or theme of finding")
    source: Source = Field(description="Source attribution for this finding")
    relevance_score: Optional[float] = Field(default=1.0, description="Relevance score (0.0 to 1.0)")

class ResearchTask(BaseModel):
    task_id: str = Field(description="Unique identifier for the task, e.g. T1, T2")
    description: str = Field(description="Actionable prompt/description of what to research")
    purpose: str = Field(default="", description="Strategic rationale for why this task is needed")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current status of the task")
    assigned_tool: str = Field(default="web_search", description="Tool selected: web_search, webpage_retrieval, rag_search, data_analysis, or calculator")
    result: Optional[str] = Field(default=None, description="Raw summary of results obtained for this task")
    error: Optional[str] = Field(default=None, description="Error message if execution failed")

class ResearchPlan(BaseModel):
    question_type: QuestionType = Field(default=QuestionType.GENERAL_BUSINESS_QUESTION, description="Determined type of business question")
    user_intent: str = Field(description="Understood intent of the user's research request")
    research_objective: str = Field(description="High-level business research objective")
    tasks: List[ResearchTask] = Field(description="Ordered list of research tasks to achieve the objective")
    recommended_report_sections: List[str] = Field(default_factory=list, description="List of report section titles appropriate for this question")

class ReportSection(BaseModel):
    title: str = Field(description="Title of the report section")
    section_type: str = Field(default="text", description="Type of section: text, table, key_metrics, bullet_list")
    content: Any = Field(description="Markdown string, list of dicts for table, dict for metrics, or list of strings")
    sources: List[str] = Field(default_factory=list, description="Cited source titles or URLs for this section")

class FinalReport(BaseModel):
    title: str = Field(default="Business Research & Intelligence Report", description="Dynamic report title based on question")
    executive_summary: str = Field(description="High-level strategic synthesis answering the user's specific question")
    question_type: QuestionType = Field(default=QuestionType.GENERAL_BUSINESS_QUESTION, description="Question classification")
    sections: List[ReportSection] = Field(default_factory=list, description="Dynamic report sections tailored to question type")
    key_takeaways: List[str] = Field(default_factory=list, description="Top actionable recommendations/takeaways")
    chart_data: Optional[Dict[str, Any]] = Field(default=None, description="Dynamic visualization structure if numerical data is present")
    gaps_and_limitations: Optional[List[str]] = Field(default=None, description="Explicit statement of genuinely unavailable or missing information, if any")
    sources: List[Source] = Field(default_factory=list, description="All cited references and evidence sources")

class ResearchState(BaseModel):
    research_question: str
    question_type: QuestionType = QuestionType.GENERAL_BUSINESS_QUESTION
    user_intent: str = ""
    research_objective: str = ""
    research_plan: Optional[ResearchPlan] = None
    tasks: List[ResearchTask] = Field(default_factory=list)
    completed_tasks: List[str] = Field(default_factory=list)
    pending_tasks: List[str] = Field(default_factory=list)
    current_task: Optional[ResearchTask] = None
    findings: List[Finding] = Field(default_factory=list)
    sources: List[Source] = Field(default_factory=list)
    tool_results: List[Dict[str, Any]] = Field(default_factory=list)
    re_research_attempts: int = 0
    errors: List[str] = Field(default_factory=list)
    final_report: Optional[FinalReport] = None
