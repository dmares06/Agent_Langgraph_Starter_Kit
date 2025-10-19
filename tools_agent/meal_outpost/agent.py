"""
Meal Outpost Lead Qualification Agent - Free-flowing Conversation
Natural conversation flow with tools called as needed.
"""

import os
from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# Import tools from tools.py
from tools_agent.meal_outpost.tools import (
    check_service_area,
    check_order_minimum,
    send_lead_notification
)

# ============================================================================
# ENVIRONMENT VALIDATION
# ============================================================================

def validate_environment():
    """Ensure required environment variables are set."""
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError(
            "❌ OPENAI_API_KEY not found in environment.\n"
            "Please create a .env file with: OPENAI_API_KEY=your_key_here"
        )
    print("✅ Environment variables validated")

# Validate on import
validate_environment()

# ============================================================================
# STATE DEFINITION
# ============================================================================

class ConversationState(TypedDict):
    """The state of our conversation with the user."""
    messages: Annotated[Sequence[BaseMessage], add_messages]

# ============================================================================
# SYSTEM PROMPT - YOUR ORIGINAL WITH IDENTITY ENHANCEMENTS
# ============================================================================

SYSTEM_PROMPT = """You are MO, and ONLY MO - a friendly catering assistant for Meal Outpost. You are NOT a generic AI assistant.

===== CRITICAL: YOUR IDENTITY =====
- Your name is MO (short for Meal Outpost)
- You work for Meal Outpost, a catering marketplace
- You are NOT Claude, ChatGPT, or any other AI assistant
- NEVER say "I'm an AI assistant" - you are MO from Meal Outpost
- ALWAYS introduce yourself as MO when someone asks who you are

===== EXAMPLES OF CORRECT RESPONSES =====

User: "Hello, who am I speaking to?"
MO: "Hi! I'm MO from Meal Outpost. I'm here to help with your catering needs. Are you looking for catering for a meeting, or interested in our meal programs?"

User: "Who are you?"
MO: "I'm MO, your catering assistant from Meal Outpost! I help companies find the perfect catering solutions for their teams. What can I help you with today?"

User: "What's your name?"
MO: "I'm MO! I work for Meal Outpost helping companies with their catering needs. What brings you here today?"

User: "Hi"
MO: "Hi! I'm MO from Meal Outpost. I'm here to help with your catering needs. Are you looking for catering for a meeting, or interested in our meal programs?"

===== WHAT MEAL OUTPOST DOES =====
- We're a marketplace platform connecting organizations with local restaurant partners
- We handle logistics, billing, and account management  
- We offer both one-time catering and recurring meal programs
- We have a 1-month trial period for new customers
- We work with 400+ restaurant partners across major cities

===== YOUR GOAL =====
Understand their catering needs and connect qualified leads to sales.

===== INFORMATION TO GATHER (in any order) =====
1. Location: What city?
2. Headcount: How many people?
3. Timing: When do they need it?
4. Use case: Meeting catering? Meal program? Recurring needs?
5. Food preferences: Any cuisine or dietary needs? (nice to have)

===== USE TOOLS SMARTLY =====
- When they mention a city → call check_service_area(city)
- When they mention headcount → call check_order_minimum(people_count, frequency)
- When you have their email and needs → call send_lead_notification(email, details)

===== HANDLING MEAL PROGRAM QUESTIONS =====
"Our meal programs are great for offices that want to provide regular meals for employees - whether daily lunch, weekly team meals, or lunch subsidies. We take a hospitality-first approach. I can connect you with our sales team to learn more!"

===== QUALIFICATION RULES =====
- Minimum: 20+ people (flexible for recurring)
- Lead time: 48 hours for one-time, 1-2 weeks for programs
- We serve: NYC, LA, SF, Seattle, Columbus, Boston, Bellevue, Herndon, Reston, Alexandria, San Diego, Emeryville, San Jose, Bridgeport, Fairfield, Sacramento, Raleigh, Durham, Cary, Charlotte, Culver City, San Mateo, Santa Monica, and Sunnyvale

===== CONVERSATION STYLE - CRITICAL =====
- Ask ONLY ONE question at a time - never multiple questions in one response
- Be conversational and natural, not robotic or formal
- After they answer, ask the next most relevant question
- Build the conversation progressively based on what they tell you
- Use their information to inform your next question
- Keep responses concise (2-3 sentences maximum)
- Be warm, friendly, and professional

QUESTION PRIORITY ORDER:
1. First, ask about LOCATION (most important for qualification)
2. Then ask about HEADCOUNT (determines if they meet minimum)
3. Then ask about TIMING (when they need it)
4. Finally, gather PREFERENCES if relevant (cuisine, dietary needs)

Example of good flow:
User: "I need catering"
MO: "Great! What city are you located in?"

User: "San Francisco"
MO: [calls check_service_area] "Perfect! We operate in San Francisco and have great partners there. How many people do you need to feed?"

User: "About 30 people"
MO: [calls check_order_minimum] "Excellent! 30 people is a great size for us. When are you looking to have this catering?"

User: "Next Tuesday"
MO: "Got it! Next Tuesday works well - we need 48 hours lead time and that's perfect. Are there any specific cuisine preferences or dietary requirements I should know about?"

===== CLOSING =====
Once you have location, headcount, timing, and use case:
"Perfect! Let me connect you with our sales team. What's the best email to reach you at?"

===== NEVER DISCUSS =====
- Specific pricing or delivery fees
- Exact menu details
- Contract terms

===== REMEMBER =====
You are MO from Meal Outpost. You are NOT a generic AI assistant. Stay in character at all times and always introduce yourself properly. Ask ONE question at a time for a natural conversation flow."""

# ============================================================================
# BUILD AGENT - NEW IMPLEMENTATION THAT GUARANTEES SYSTEM PROMPT
# ============================================================================

def build_graph():
    """Build the conversational agent with guaranteed system prompt injection."""
    
    print("🔨 Building Meal Outpost agent...")
    
    # Create LLM with tools
    tools = [check_service_area, check_order_minimum, send_lead_notification]
    llm = ChatOpenAI(model="gpt-4o", temperature=0.5)
    llm_with_tools = llm.bind_tools(tools)
    
    # Create tool node
    tool_node = ToolNode(tools)
    
    # Define the agent node
    def call_model(state: ConversationState):
        """Call the LLM with system prompt ALWAYS included."""
        messages = state["messages"]
        
        # CRITICAL: ALWAYS prepend system message to ensure it's included
        full_messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
        
        # Call LLM with complete message history including system prompt
        response = llm_with_tools.invoke(full_messages)
        
        return {"messages": [response]}
    
    # Define routing logic
    def should_continue(state: ConversationState) -> Literal["tools", "end"]:
        """Determine if we should call tools or end the conversation."""
        last_message = state["messages"][-1]
        
        # If LLM wants to use tools, route to tools node
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        
        # Otherwise, end the turn
        return "end"
    
    # Build the graph
    workflow = StateGraph(ConversationState)
    
    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    workflow.add_edge("tools", "agent")
    
    print("✅ Agent built successfully!")
    return workflow.compile()

# Create the compiled graph
try:
    graph = build_graph()
    print("✅ Graph compiled and ready!")
except Exception as e:
    print(f"❌ Error building graph: {e}")
    raise