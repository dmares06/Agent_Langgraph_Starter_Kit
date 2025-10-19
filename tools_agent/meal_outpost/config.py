# tools_agent/meal_outpost/config.py
import os
from typing import List
from pydantic import BaseModel, Field

class AgentConfig(BaseModel):
    """Configuration for the Meal Outpost agent."""
    
    # Model configuration  
    model_name: str = "gpt-4o"
    temperature: float = 0.7
    
    # Business rules
    minimum_order_size: int = 20
    lead_time_hours: int = 48
    sales_email: str = "dylan@mealoutpost.com"
    
    # API keys from environment
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))

# Global config instance
config = AgentConfig()

def validate_config():
    """Validate configuration is complete."""
    issues = []
    
    if not config.openai_api_key:
        issues.append("No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
    
    if issues:
        raise ValueError("Configuration issues:\n" + "\n".join(f"- {issue}" for issue in issues))