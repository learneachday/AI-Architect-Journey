def calculator(expression: str) -> str:
    """Safe calculator tool"""
    try:
        # Safe eval for basic math only
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Calculation error: {str(e)}"
