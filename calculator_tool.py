import math
import re
from typing import Dict, Any

def execute_calculation(expression: str) -> Dict[str, Any]:
    """
    Safely evaluates a mathematical or numeric calculation expression.
    Supports basic arithmetic, percentages, growth rates, averages, and standard math functions.
    """
    clean_expr = expression.strip()
    
    # Try extracting mathematical expression from text description
    math_match = re.search(r'([\d\.\s\+\-\*\/\(\)\^\%]+)', clean_expr)
    
    # Safe allowed evaluation namespace
    allowed_names = {
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'sum': sum,
        'pow': pow,
        'sqrt': math.sqrt,
        'log': math.log,
        'exp': math.exp,
        'pi': math.pi,
        'e': math.e
    }
    
    # Prepare expression for evaluation
    eval_expr = clean_expr
    eval_expr = eval_expr.replace('^', '**')
    eval_expr = re.sub(r'(\d+)%', r'(\1/100)', eval_expr)

    # Sanitize string: ensure only numbers, math operators, allowed identifiers, commas, parens
    sanitized = re.sub(r'[^0-9\+\-\*\/\(\)\.\,\s\_a-zA-Z]', '', eval_expr)
    
    try:
        # Check for non-allowed tokens
        compiled = compile(sanitized, "<string>", "eval")
        for name in compiled.co_names:
            if name not in allowed_names:
                raise NameError(f"Use of '{name}' is not allowed in calculator.")
        
        result = eval(compiled, {"__builtins__": None}, allowed_names)
        
        if isinstance(result, float):
            result_formatted = f"{result:,.4f}".rstrip('0').rstrip('.')
        else:
            result_formatted = f"{result:,}"

        return {
            "status": "success",
            "expression": expression,
            "evaluated_expression": sanitized,
            "numeric_result": result,
            "formatted_result": result_formatted,
            "summary": f"Calculated result for '{expression}': {result_formatted}"
        }
    except Exception as e:
        return {
            "status": "error",
            "expression": expression,
            "message": f"Calculation error for '{expression}': {str(e)}",
            "summary": f"Failed to calculate expression '{expression}': {str(e)}"
        }
