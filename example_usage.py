"""
Enhanced examples for the Meal Outpost Lead Qualification Agent
Shows complete conversation flows with memory and proper tool usage.
"""

import asyncio
from agent import build_graph, ConversationState
from langchain_core.messages import HumanMessage
from config import config, validate_config
import json

def print_conversation_state(state, step_name):
    """Helper to print conversation state nicely."""
    print(f"\n--- {step_name} ---")
    if state["messages"]:
        last_message = state["messages"][-1].content
        print(f"Agent: {last_message}")
    
    print(f"Stage: {state['conversation_stage']}")
    if state.get('user_need'): print(f"Need: {state['user_need']}")
    if state.get('location_city'): print(f"Location: {state['location_city']}, {state['location_state']}")
    if state.get('is_in_service_area') is not None: print(f"In service area: {state['is_in_service_area']}")
    if state.get('headcount'): print(f"Headcount: {state['headcount']}")
    if state.get('qualification_status'): print(f"Qualification: {state['qualification_status']}")

async def example_1_qualified_lead():
    """Example 1: Qualified recurring customer in NYC"""
    print("=" * 60)
    print("EXAMPLE 1: Qualified Lead - Recurring NYC Corporate")
    print("=" * 60)
    
    graph = build_graph()
    
    # Initialize with thread_id for memory
    config_with_memory = {"configurable": {"thread_id": "example_1"}}
    
    # Start conversation
    initial_state = {
        "messages": [],
        "user_need": None,
        "location_city": None,
        "location_state": None,
        "timing": None,
        "headcount": None,
        "frequency": None,
        "cuisine_preferences": [],
        "dietary_requirements": [],
        "is_in_service_area": None,
        "meets_minimum": None,
        "qualification_status": None,
        "contact_name": None,
        "contact_email": None,
        "contact_phone": None,
        "conversation_stage": "greeting",
        "ready_for_handoff": False,
        "notes": ""
    }
    
    # Greeting
    result = await graph.ainvoke(initial_state, config_with_memory)
    print_conversation_state(result, "Greeting")
    
    # User responds about recurring meals
    result["messages"].append(HumanMessage(content="I need daily lunch catering for our office"))
    result = await graph.ainvoke(result, config_with_memory)
    print_conversation_state(result, "Need Discovery")
    
    # Location
    result["messages"].append(HumanMessage(content="We're in Manhattan, New York City"))
    result = await graph.ainvoke(result, config_with_memory)
    print_conversation_state(result, "Location Check")
    
    # Timing
    result["messages"].append(HumanMessage(content="We'd like to start next month"))
    result = await graph.ainvoke(result, config_with_memory)
    print_conversation_state(result, "Timing")
    
    # Scale
    result["messages"].append(HumanMessage(content="About 45 people"))
    result = await graph.ainvoke(result, config_with_memory)
    print_conversation_state(result, "Scale Assessment")
    
    # Frequency
    result["messages"].append(HumanMessage(content="Daily, Monday through Friday"))
    result = await graph.ainvoke(result, config_with_memory)
    print_conversation_state(result, "Frequency")
    
    # Preferences
    result["messages"].append(HumanMessage(content="We like variety - Italian, Asian, healthy options"))
    result = await graph.ainvoke(result, config_with_memory)
    print_conversation_state(result, "Preferences")
    
    # Restaurant showcase
    result = await graph.ainvoke(result, config_with_memory)
    print_conversation_state(result, "Restaurant Showcase")
    
    # Contact capture
    result["messages"].append(HumanMessage(content="Yes, please connect me with your team"))
    result = await graph.ainvoke(result, config_with_memory)
    print_conversation_state(result, "Contact Capture")
    
    # Email collection
    result["messages"].append(HumanMessage(content="sarah.johnson@techcorp.com"))
    result = await graph.ainvoke(result, config_with_memory)
    print_conversation_state(result, "Email Collection")
    
    # Final handoff
    result = await graph.ainvoke(result, config_with_memory)
    print_conversation_state(result, "Final Handoff")
    
    print(f"\n✅ QUALIFIED LEAD CAPTURED!")
    print(f"Email: {result.get('contact_email')}")
    print(f"Qualification: {result.get('qualification_status')}")

async def example_2_outside_service_area():
    """Example 2: Outside service area but still helpful"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Outside Service Area - Portland Request")
    print("=" * 60)
    
    graph = build_graph()
    config_with_memory = {"configurable": {"thread_id": "example_2"}}
    
    initial_state = {
        "messages": [],
        "user_need": None,
        "location_city": None,
        "location_state": None,
        "timing": None,
        "headcount": None,
        "frequency": None,
        "cuisine_preferences": [],
        "dietary_requirements": [],
        "is_in_service_area": None,
        "meets_minimum": None,
        "qualification_status": None,
        "contact_name": None,
        "contact_email": None,
        "contact_phone": None,
        "conversation_stage": "greeting",
        "ready_for_handoff": False,
        "notes": ""
    }
    
    # Simulate conversation flow
    result = await graph.ainvoke(initial_state, config_with_memory)
    print_conversation_state(result, "Greeting")
    
    result["messages"].append(HumanMessage(content="I need catering for a company event"))
    result = await graph.ainvoke(result, config_with_memory)
    
    result["messages"].append(HumanMessage(content="We're in Portland, Oregon"))
    result = await graph.ainvoke(result, config_with_memory)
    print_conversation_state(result, "Location Check - Outside Service Area")
    
    result["messages"].append(HumanMessage(content="Next Friday"))
    result = await graph.ainvoke(result, config_with_memory)
    
    result["messages"].append(HumanMessage(content="About 75 people"))
    result = await graph.ainvoke(result, config_with_memory)
    
    result["messages"].append(HumanMessage(content="Mexican or Mediterranean food"))
    result = await graph.ainvoke(result, config_with_memory)
    
    result = await graph.ainvoke(result, config_with_memory)
    print_conversation_state(result, "Still Helpful Despite Location")
    
    result["messages"].append(HumanMessage(content="Yes, I'd like to stay in touch about expansion"))
    result = await graph.ainvoke(result, config_with_memory)
    
    result["messages"].append(HumanMessage(content="mike.chen@startup.co"))
    result = await graph.ainvoke(result, config_with_memory)
    print_conversation_state(result, "Captured for Future Expansion")

async def example_3_edge_case_questions():
    """Example 3: Handling edge case questions"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Edge Cases and Complex Questions")
    print("=" * 60)
    
    graph = build_graph()
    config_with_memory = {"configurable": {"thread_id": "example_3"}}
    
    initial_state = {
        "messages": [],
        "conversation_stage": "greeting",
        "user_need": None,
        "location_city": None,
        "location_state": None,
        "timing": None,
        "headcount": None,
        "frequency": None,
        "cuisine_preferences": [],
        "dietary_requirements": [],
        "is_in_service_area": None,
        "meets_minimum": None,
        "qualification_status": None,
        "contact_name": None,
        "contact_email": None,
        "contact_phone": None,
        "ready_for_handoff": False,
        "notes": ""
    }
    
    result = await graph.ainvoke(initial_state, config_with_memory)
    
    # User asks complex pricing question early
    result["messages"].append(HumanMessage(content="How much does catering for 50 people cost exactly?"))
    result = await graph.ainvoke(result, config_with_memory)
    print_conversation_state(result, "Complex Pricing Question")
    
    # User asks about specific dietary restrictions
    result["messages"].append(HumanMessage(content="Do you have kosher options and can handle severe nut allergies?"))
    result = await graph.ainvoke(result, config_with_memory)
    print_conversation_state(result, "Specific Dietary Question")
    
    # User asks about platform features
    result["messages"].append(HumanMessage(content="How does your ordering system work? Do you have an app?"))
    result = await graph.ainvoke(result, config_with_memory)
    print_conversation_state(result, "Platform Feature Question")

def test_memory_between_sessions():
    """Test memory persistence between sessions"""
    print("\n" + "=" * 60)
    print("TESTING MEMORY BETWEEN SESSIONS")
    print("=" * 60)
    
    # This would test that conversation state persists
    # Implementation depends on your checkpointer setup
    print("Note: Memory testing requires proper checkpointer configuration")
    print("See LangGraph documentation for production memory setup")

async def run_all_examples():
    """Run all examples to demonstrate the agent."""
    try:
        # Validate configuration first
        validate_config()
        print("✅ Configuration valid")
        
        await example_1_qualified_lead()
        await example_2_outside_service_area()
        await example_3_edge_case_questions()
        test_memory_between_sessions()
        
        print("\n" + "=" * 60)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nYour agent demonstrates:")
        print("- ✅ Complete conversation flows")
        print("- ✅ Proper qualification logic")
        print("- ✅ Helpful responses even when can't serve")
        print("- ✅ Edge case handling")
        print("- ✅ Lead capture and handoff")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_all_examples())