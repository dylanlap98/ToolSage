"""
Inline dataset and task definition for the data analyst demo.
No external files — everything the agent needs is in SALES_DATA.
"""

SALES_DATA = {
    "months": ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    "revenue": [42000, 38500, 51200, 47800, 63400, 58900,
                71200, 68500, 55300, 61800, 78400, 92100],
    "units_sold": [210, 185, 248, 231, 312, 287,
                   356, 334, 269, 301, 389, 461],
    "region": "North America",
    "year": 2024,
}

ANALYSIS_TASK = """\
Analyze the 2024 North America sales data and answer each of the following:

1. Which month had the highest revenue? Which had the lowest?
2. What was the total annual revenue and the average monthly revenue?
3. Calculate the month-over-month revenue growth rate for each month (starting Feb).
4. Which quarter performed best by total revenue? (Q1=Jan-Mar, Q2=Apr-Jun, Q3=Jul-Sep, Q4=Oct-Dec)
5. Is there a correlation between units sold and revenue? Show the correlation coefficient.

Use the python_repl tool for each question separately. Print clear, formatted answers.\
"""
