import os
import pandas as pd
from typing import Dict, Any, List

def analyze_tabular_file(file_path: str, user_query: str) -> Dict[str, Any]:
    """
    Analyzes CSV or Excel files using Pandas for structured data inspection and metrics calculation.
    """
    if not os.path.exists(file_path):
        return {
            "status": "error",
            "file_name": os.path.basename(file_path),
            "message": f"File not found: {file_path}"
        }

    file_name = os.path.basename(file_path)
    ext = os.path.splitext(file_name)[1].lower()

    try:
        if ext == ".csv":
            df = pd.read_csv(file_path)
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)
        else:
            return {
                "status": "error",
                "file_name": file_name,
                "message": f"Unsupported tabular format: {ext}"
            }

        shape = df.shape
        columns = list(df.columns)
        
        # Determine numeric vs categorical
        numeric_cols = list(df.select_dtypes(include=['number']).columns)
        categorical_cols = [col for col in columns if col not in numeric_cols]

        # Calculate summary statistics
        summary_stats = {}
        if numeric_cols:
            desc = df[numeric_cols].describe().to_dict()
            for col, stats in desc.items():
                summary_stats[col] = {
                    "count": int(stats.get("count", 0)),
                    "mean": float(round(stats.get("mean", 0), 2)),
                    "std": float(round(stats.get("std", 0), 2)),
                    "min": float(round(stats.get("min", 0), 2)),
                    "max": float(round(stats.get("max", 0), 2)),
                    "sum": float(round(df[col].sum(), 2))
                }

        top_categories = {}
        for col in categorical_cols[:5]:
            top_categories[col] = df[col].value_counts().head(5).to_dict()

        head_sample = df.head(10).to_dict(orient="records")

        # Specific analysis based on user query keywords
        query_analysis = []
        q_lower = user_query.lower()

        if any(w in q_lower for w in ["highest", "top", "max", "rank", "best", "leading"]):
            for num_col in numeric_cols:
                try:
                    max_idx = df[num_col].idxmax()
                    max_val = df.loc[max_idx, num_col]
                    row_info = {k: v for k, v in df.loc[max_idx].to_dict().items() if pd.notna(v)}
                    query_analysis.append(f"Highest '{num_col}' value is {max_val} in row: {row_info}")
                except Exception:
                    pass

        if any(w in q_lower for w in ["lowest", "min", "bottom", "worst"]):
            for num_col in numeric_cols:
                try:
                    min_idx = df[num_col].idxmin()
                    min_val = df.loc[min_idx, num_col]
                    row_info = {k: v for k, v in df.loc[min_idx].to_dict().items() if pd.notna(v)}
                    query_analysis.append(f"Lowest '{num_col}' value is {min_val} in row: {row_info}")
                except Exception:
                    pass

        if any(w in q_lower for w in ["total", "sum", "overall", "aggregate", "revenue", "sales"]):
            for num_col in numeric_cols:
                total_val = df[num_col].sum()
                query_analysis.append(f"Total sum of '{num_col}': {total_val:,.2f}")

        if any(w in q_lower for w in ["average", "mean", "avg"]):
            for num_col in numeric_cols:
                avg_val = df[num_col].mean()
                query_analysis.append(f"Average of '{num_col}': {avg_val:,.2f}")

        # Perform groupby if category + numeric exist
        groupby_insights = []
        if categorical_cols and numeric_cols:
            primary_cat = categorical_cols[0]
            primary_num = numeric_cols[0]
            try:
                grouped = df.groupby(primary_cat)[primary_num].agg(['sum', 'mean', 'count']).sort_values(by='sum', ascending=False).head(5)
                for cat_val, grp_row in grouped.iterrows():
                    groupby_insights.append(f"Group '{cat_val}': Total {primary_num} = {grp_row['sum']:,.2f}, Avg = {grp_row['mean']:,.2f}, Count = {int(grp_row['count'])}")
            except Exception:
                pass

        analysis_text = f"Tabular Dataset '{file_name}' Summary:\n"
        analysis_text += f"- Dataset Dimensions: {shape[0]} rows x {shape[1]} columns\n"
        analysis_text += f"- Columns: {', '.join(columns)}\n"
        if summary_stats:
            analysis_text += f"- Numeric Summary Statistics: {summary_stats}\n"
        if query_analysis:
            analysis_text += "- Query-Specific Observations:\n  " + "\n  ".join(query_analysis) + "\n"
        if groupby_insights:
            analysis_text += "- Top Aggregations:\n  " + "\n  ".join(groupby_insights) + "\n"
        analysis_text += f"- Sample Records: {head_sample[:3]}"

        return {
            "status": "success",
            "file_name": file_name,
            "shape": shape,
            "columns": columns,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "summary_stats": summary_stats,
            "top_categories": top_categories,
            "query_analysis": query_analysis,
            "groupby_insights": groupby_insights,
            "head_sample": head_sample,
            "analysis_text": analysis_text
        }

    except Exception as e:
        return {
            "status": "error",
            "file_name": file_name,
            "message": f"Error analyzing spreadsheet with Pandas: {str(e)}"
        }
