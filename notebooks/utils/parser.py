def parse_currency(value):
    """
    Converts currency string to float, handling various formats.

    Args:
        value (str or numeric): Value to convert

    Returns:
        float: Parsed numeric value
    """
    if isinstance(value, (int, float)):
        return value

    return float(value.replace(",", "").replace("USD", "").replace("$", "").strip())
