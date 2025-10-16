"""
Meal Outpost Lead Qualification Agent
A conversational agent that helps visitors determine if Meal Outpost can serve their catering needs.
"""

from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import json

# ============================================================================
# STATE DEFINITION
# ============================================================================

class ConversationState(TypedDict):
    """The state of our conversation with the user."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # Information we're gathering
    user_need: str | None  # meeting, recurring, exploring
    location_city: str | None
    location_state: str | None
    timing: str | None  # when they need service
    headcount: int | None
    frequency: str | None  # one-time, daily, weekly, monthly
    cuisine_preferences: list[str]
    dietary_requirements: list[str]
    
    # Qualification status
    is_in_service_area: bool | None
    meets_minimum: bool | None
    qualification_status: str | None  # qualified, maybe, not-qualified
    
    # Contact information (collected at end)
    contact_name: str | None
    contact_email: str | None
    contact_phone: str | None
    
    # Flow control
    conversation_stage: str  # greeting, discovery, location, etc.
    ready_for_handoff: bool
    notes: str  # Any special requirements or context


# ============================================================================
# CONFIGURATION
# ============================================================================

class GraphConfigPydantic(BaseModel):
    """Configuration for the Meal Outpost agent."""
    
    model_name: str = Field(
        default="gpt-4o",
        description="The LLM model to use for the agent"
    )
    
    temperature: float = Field(
        default=0.7,
        description="Temperature for model responses (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    
    sales_email: str = Field(
        default="sales@mealoutpost.com",
        description="Email for sales team notifications"
    )
    
    minimum_order_size: int = Field(
        default=20,
        description="Minimum number of people per order to be considered qualified"
    )


# ============================================================================
# SYSTEM PROMPT
# ============================================================================

SYSTEM_PROMPT = """You are a helpful assistant for Meal Outpost, a catering marketplace that connects businesses, universities, government buildings, sports teams, and other organizations with restaurant partners for their food needs.

WHAT MEAL OUTPOST DOES:
- We're a marketplace platform (not a single restaurant)
- We connect organizations with local restaurant partners for catering
- We handle logistics, billing, and account management
- We offer both one-time catering and recurring meal programs
- We offer a 1 month trial period for all new customers who want to try our service.
- We work with 400+ restaurant partners across 18 major cities

YOUR ROLE:
- Help visitors understand if Meal Outpost can serve their catering needs
- Gather information about their requirements in a friendly, conversational way
- Stay helpful even if they're outside our current service area or below minimums
- Showcase relevant restaurant partners to demonstrate our value
- Connect qualified leads with our sales team

CONVERSATION GUIDELINES:
- Be warm, professional, and helpful
- Ask ONE question at a time - don't overwhelm
- Stay helpful even if they can't be served immediately
- Guide conversation: need → location → scale → preferences → restaurant showcase
- For edge cases, say: "I don't have that specific information, but our team at sales@mealoutpost.com can help with [specific question]"
- Remember: we prefer 20+ people per order, but still capture smaller leads for future opportunities

CONVERSATION FLOW:
- Start with understanding their need (one-time event, recurring program, exploring)
- Ask about location to check service area
- Understand timing and scale (how many people, when needed)
- Gather preferences (cuisine types, dietary needs)
- Showcase relevant restaurant partners
- Capture contact info for qualified leads
- Always complete the flow even if they can't be served immediately

SERVICE AREAS: (cities we operate in)
- New York City, Los Angeles, , San Francisco, Boston, Washington DC, Seattle, Herndon VA, Reston, VA, Alexandria VA, Columbus OH, San Diego, Emeryville, San Jose

MINIMUM REQUIREMENTS: 
- Preferred: 20+ people per order
- We work with smaller orders if they have recurring needs or multiple events planned
- Lead time: 48 hours for one-time, 1-2 weeks for program setup

QUALIFICATION APPROACH:
- Soft qualification - gather info and set expectations rather than rejecting
- Even outside service area: "We're expanding and would love to stay in touch"
- Even below minimums: "We typically work best with larger orders, but we would be happy to discuss your needs"

RESTAURANT PARTNERS:
-Show 3 - 4 specific restauraunts that are in the area they are looking for to give them an idea of the quality of the food we can provide.
- Mention an estimate # of restaurant parnters in their area 
- Focus on restaurants that match the cuisine preferences they have shared.
- Example: 'In Columbus, we work with 40+ restaurants including Alchemy, Bibibop, Hot Chicken Takeover, and more.'

WHAT YOU KNOW ABOUT:
- Service areas (cities we operate in)
- Restaurant partners in each location
- Lead time requirements
- How our platform works
- Types of programs we offer (one-time catering, recurring meal programs)

WHAT TO AVOID DISCUSSING:
- Specific pricing or delivery fees
- Exact menu costs
- Contract terms

For pricing questions, say: "That's a great question about pricing. Our sales team would be happy to help - just reach out to dylan@mealoutpost.com and we can provide more detailed information based on your specific needs."
"""


# ============================================================================
# AGENT NODES
# ============================================================================

def greeting_node(state: ConversationState) -> ConversationState:
    """Initial greeting and understanding what they need."""
    
    if state["conversation_stage"] != "greeting":
        return state
    
    # This is the initial greeting
    greeting = """Hi! Welcome to Meal Outpost. I'm here to help you figure out if we can support your catering needs.

What brings you to Meal Outpost today?

Are you looking for:
• One-time catering for a meeting or event?
• A recurring meal program (daily/weekly lunches)?
• Just exploring your options?"""

    state["messages"].append(AIMessage(content=greeting))
    state["conversation_stage"] = "discovery"
    
    return state


def discovery_node(state: ConversationState) -> ConversationState:
    """Understand their specific needs and move to location."""
    
    if state["conversation_stage"] != "discovery":
        return state
    
    # Use LLM to analyze their response and extract need type
    last_message = state["messages"][-1].content
    
    # Simple keyword detection (in production, use LLM to extract)
    if any(word in last_message.lower() for word in ["meeting", "event", "one-time", "once"]):
        state["user_need"] = "one-time"
    elif any(word in last_message.lower() for word in ["recurring", "daily", "weekly", "program", "regular"]):
        state["user_need"] = "recurring"
    else:
        state["user_need"] = "exploring"
    
    # Ask about location
    response = "Great! Where are you looking to get catering? What city are you in?"
    state["messages"].append(AIMessage(content=response))
    state["conversation_stage"] = "location"
    
    return state


def location_node(state: ConversationState) -> ConversationState:
    """Check if we serve their location."""
    
    if state["conversation_stage"] != "location":
        return state
    
    # Extract location from last message
    # In production, use LLM/NER to extract city/state
    last_message = state["messages"][-1].content.lower()
    
    # Service areas
    service_cities = {
        "new york": "NY", "nyc": "NY", "new york city": "NY",
        "los angeles": "CA", "la": "CA",
        "bellevue": "WA", "bellevue wa": "WA",
        "san francisco": "CA", "sf": "CA",
        "boston": "MA", "bos": "MA",
        "washington": "DC", "dc": "DC",
        "seattle": "WA", "sea": "WA",
        "herndon": "VA", "herndon va": "VA",
        "reston": "VA", "reston va": "VA",
        "alexandria": "VA", "alexandria va": "VA",  
        "columbus": "OH", "columbus oh": "OH",
        "san diego": "CA", "san diego ca": "CA",
        "emeryville": "CA", "emeryville ca": "CA",
        "san jose": "CA", "san jose ca": "CA",
        "bridgeport": "CT", "bridgeport ct": "CT",
        "fairfield": "CT", "fairfield ct": "CT",
        "Sacramento": "CA", "sacramento ca": "CA",
        "raleigh": "NC", "raleigh nc": "NC",
        "durham": "NC", "durham nc": "NC",
        "cary": "NC", "cary nc": "NC",
        "charlotte": "NC", "charlotte nc": "NC",
        "culver city": "CA", "culver city ca": "CA",
        "san mateo": "CA", "san mateo ca": "CA",
        "santa monica": "CA", "santa monica ca": "CA",
        "sunnyvale": "CA", "sunnyvale ca": "CA"
    }
    
    city_found = None
    for city, state_abbr in service_cities.items():
        if city in last_message:
            state["location_city"] = city.title()
            state["location_state"] = state_abbr
            state["is_in_service_area"] = True
            city_found = city.title()
            break
    
    if not city_found:
        state["is_in_service_area"] = False
        response = f"""I appreciate you sharing that! We don't currently serve that area, but I'd still love to learn more about your needs. We're always expanding, and I can connect you with our team.

When are you looking to have catering?"""
    else:
        response = f"""Perfect! We serve {city_found} and have great restaurant partners there.

When are you looking to have catering?"""
    
    state["messages"].append(AIMessage(content=response))
    state["conversation_stage"] = "timing"
    
    return state


def timing_node(state: ConversationState) -> ConversationState:
    """Understand when they need service."""
    
    if state["conversation_stage"] != "timing":
        return state
    
    last_message = state["messages"][-1].content
    state["timing"] = last_message  # Store their timing info
    
    response = "Got it! How many people are you looking to feed?"
    state["messages"].append(AIMessage(content=response))
    state["conversation_stage"] = "scale"
    
    return state


def scale_node(state: ConversationState) -> ConversationState:
    """Gather headcount and frequency information."""
    
    if state["conversation_stage"] != "scale":
        return state
    
    last_message = state["messages"][-1].content
    
    # Extract number (simple version - use LLM in production)
    import re
    numbers = re.findall(r'\d+', last_message)
    if numbers:
        state["headcount"] = int(numbers[0])
        state["meets_minimum"] = state["headcount"] >= 20
    
    # Ask about frequency
    if state["user_need"] == "recurring":
        response = "And how often would you need catering? Daily, weekly, or something else?"
        state["conversation_stage"] = "frequency"
    else:
        response = "Thanks! Do you have any cuisine preferences or dietary requirements? (e.g., Italian, vegetarian, gluten-free)"
        state["conversation_stage"] = "preferences"
    
    state["messages"].append(AIMessage(content=response))
    return state


def frequency_node(state: ConversationState) -> ConversationState:
    """For recurring orders, understand frequency."""
    
    if state["conversation_stage"] != "frequency":
        return state
    
    state["frequency"] = state["messages"][-1].content
    
    response = "Perfect! Do you have any cuisine preferences or dietary requirements? (e.g., Italian, vegetarian, gluten-free)"
    state["messages"].append(AIMessage(content=response))
    state["conversation_stage"] = "preferences"
    
    return state


def preferences_node(state: ConversationState) -> ConversationState:
    """Gather food preferences."""
    
    if state["conversation_stage"] != "preferences":
        return state
    
    prefs = state["messages"][-1].content
    # Simple extraction (use LLM in production)
    state["cuisine_preferences"] = [prefs]
    
    state["conversation_stage"] = "restaurant_partners"
    return state


def get_restaurant_partners_node(state: ConversationState) -> ConversationState:
    """Get relevant restaurant partners."""
    
    if state["conversation_stage"] != "restaurant_partners":
        return state
    
    if state["is_in_service_area"]:
        response = f"""Great choices! In {state["location_city"]}, we work with some excellent restaurant partners that could be perfect for you:

• **Ben's Fast Food** - Healthy delicious bowls with a variety of options
• **Pokeworks** - Incredible poke bowls with a variety of options
• **Starbird Chicken** - Delicious chicken meals 

Plus many other partners in your area! We can match you with the perfect fit for your needs.

Would you like me to connect you with our team who can provide personalized recommendations and pricing?"""
    else:
        response = """Based on what you've shared, this sounds like a great fit for the type of service we provide! Even though we don't serve your area yet, we're always expanding.

Would you like me to connect you with our team? They can discuss potential options or keep you updated on when we expand to your region."""
    
    state["messages"].append(AIMessage(content=response))
    state["conversation_stage"] = "contact_capture"
    
    return state


def contact_capture_node(state: ConversationState) -> ConversationState:
    """Get contact information for handoff."""
    
    if state["conversation_stage"] != "contact_capture":
        return state
    
    last_message = state["messages"][-1].content.lower()
    
    if "yes" in last_message or "sure" in last_message or "please" in last_message:
        response = """Perfect! What's the best email address to reach you at?

(Our team typically responds within a few hours during business hours)"""
        state["messages"].append(AIMessage(content=response))
        state["conversation_stage"] = "email_collection"
    else:
        response = """No problem! If you change your mind, you can always reach our team at sales@mealoutpost.com.

Is there anything else I can help you understand about Meal Outpost?"""
        state["messages"].append(AIMessage(content=response))
        state["conversation_stage"] = "end"
    
    return state


def email_collection_node(state: ConversationState) -> ConversationState:
    """Collect email and send notification to sales team."""
    
    if state["conversation_stage"] != "email_collection":
        return state
    
    email = state["messages"][-1].content.strip()
    state["contact_email"] = email
    
    # Determine qualification status
    if state["is_in_service_area"] and state["meets_minimum"]:
        qual_status = "Qualified"
    elif state["is_in_service_area"] or (state["headcount"] and state["headcount"] >= 10):
        qual_status = "Maybe"
    else:
        qual_status = "Not Qualified"
    
    state["qualification_status"] = qual_status
    state["ready_for_handoff"] = True
    
    # Send notification to sales team (no database storage)
    from .tools import send_notification
    
    # Create lead data for email notification only
    notification_data = {
        "contact": {"email": email},
        "requirements": {
            "location": {"city": state["location_city"], "state": state["location_state"]},
            "need_type": state["user_need"],
            "headcount": state["headcount"],
            "timing": state["timing"],
            "frequency": state.get("frequency"),
            "cuisine_preferences": state["cuisine_preferences"]
        },
        "qualification": {
            "status": qual_status,
            "in_service_area": state["is_in_service_area"],
            "meets_minimum": state["meets_minimum"]
        }
    }
    
    # Send email notification (not stored anywhere)
    send_notification("temp-lead-id", notification_data, "general_sales")
    
    response = f"""Thanks, {email}! I've notified our team about your catering needs.

Here's a quick summary:
- Location: {state["location_city"] or "Outside current service area"}
- Need: {state["user_need"]}
- Group size: {state["headcount"]} people
- Timing: {state["timing"]}

Someone from our team will reach out to you within 24 hours!

Is there anything else you'd like to know in the meantime?"""
    
    state["messages"].append(AIMessage(content=response))
    state["conversation_stage"] = "handoff"
    
    return state


def handoff_node(state: ConversationState) -> ConversationState:
    """Final handoff - create lead and notify team."""
    
    if state["conversation_stage"] != "handoff":
        return state
    
    # In production, this would call create_lead tool
    # For now, just mark as complete
    
    response = """Perfect! Thanks so much for chatting with me today. Our team has all your details and will be in touch soon.

Have a great day! 🎉"""
    
    state["messages"].append(AIMessage(content=response))
    state["conversation_stage"] = "complete"
    
    return state


# ============================================================================
# ROUTING LOGIC
# ============================================================================

def route_conversation(state: ConversationState) -> str:
    """Determine which node to go to next based on conversation stage."""
    
    stage = state["conversation_stage"]
    
    stage_map = {
        "greeting": "discovery",
        "discovery": "location",
        "location": "timing",
        "timing": "scale",
        "scale": "frequency" if state.get("user_need") == "recurring" else "preferences",
        "frequency": "preferences",
        "preferences": "restaurant_partners",
        "restaurant_showcase": "contact_capture",
        "contact_capture": "email_collection",
        "email_collection": "handoff",
        "handoff": END,
        "complete": END,
        "end": END
    }
    
    return stage_map.get(stage, END)


# ============================================================================
# BUILD GRAPH
# ============================================================================

def build_graph():
    """Construct the LangGraph state machine."""
    
    workflow = StateGraph(ConversationState)
    
    # Add all nodes
    workflow.add_node("greeting", greeting_node)
    workflow.add_node("discovery", discovery_node)
    workflow.add_node("location", location_node)
    workflow.add_node("timing", timing_node)
    workflow.add_node("scale", scale_node)
    workflow.add_node("frequency", frequency_node)
    workflow.add_node("preferences", preferences_node)
    workflow.add_node("restaurant_partners", get_restaurant_partners_node)
    workflow.add_node("contact_capture", contact_capture_node)
    workflow.add_node("email_collection", email_collection_node)
    workflow.add_node("handoff", handoff_node)
    
    # Set entry point
    workflow.set_entry_point("greeting")
    
    # Add conditional edges based on conversation stage
    workflow.add_conditional_edges(
        "greeting",
        route_conversation
    )
    workflow.add_conditional_edges(
        "discovery",
        route_conversation
    )
    workflow.add_conditional_edges(
        "location",
        route_conversation
    )
    workflow.add_conditional_edges(
        "timing",
        route_conversation
    )
    workflow.add_conditional_edges(
        "scale",
        route_conversation
    )
    workflow.add_conditional_edges(
        "frequency",
        route_conversation
    )
    workflow.add_conditional_edges(
        "preferences",
        route_conversation
    )
    workflow.add_conditional_edges(
        "restaurant_partners",
        route_conversation
    )
    workflow.add_conditional_edges(
        "contact_capture",
        route_conversation
    )
    workflow.add_conditional_edges(
        "email_collection",
        route_conversation
    )
    workflow.add_conditional_edges(
        "handoff",
        route_conversation
    )
    
    return workflow.compile()


# ============================================================================
# MAIN AGENT
# ============================================================================

# Create the compiled graph
graph = build_graph()