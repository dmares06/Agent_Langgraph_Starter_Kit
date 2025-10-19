"""
Meal Outpost Lead Qualification Agent
A conversational agent for qualifying catering leads.
"""

__version__ = "0.1.0"

# This allows imports like: from tools_agent.meal_outpost import graph
from tools_agent.meal_outpost.agent import graph

__all__ = ["graph", "__version__"]