"""
Test script for the Meal Outpost Lead Qualification Agent
Run this to test your agent locally before deploying.
"""

from agent import build_graph, ConversationState
from tools import (
    check_service_area,
    get_restaurant_partners,
    calculate_lead_score
)
from langchain_core.messages import HumanMessage

def test_service_area_check():
    """Test service area checking function."""
    print("\n=== Testing Service Area Check ===")
    
    # Test existing city
    result = check_service_area("New York City", "NY")
    print(f"NYC Result: {result}")
    assert result['serves_area'] == True
    
    # Test non-existent city
    result = check_service_area("Small Town", "TX")
    print(f"Small Town Result: {result}")
    assert result['serves_area'] == False
    
    print("✅ Service area check tests passed!")


def test_restaurant_partners():
    """Test restaurant partner retrieval."""
    print("\n=== Testing Restaurant Partners ===")
    
    # Get restaurants in NYC
    restaurants = get_restaurant_partners("New York City", limit=3)
    print(f"Found {len(restaurants)} restaurants in NYC:")
    for r in restaurants:
        print(f"  - {r['name']}: {r['cuisine']}")
    
    # Get Italian restaurants
    italian = get_restaurant_partners(
        "New York City",
        cuisine_type=["Italian"],
        limit=2
    )
    print(f"\nFound {len(italian)} Italian restaurants:")
    for r in italian:
        print(f"  - {r['name']}")
    
    # Get restaurants with dietary options
    vegan = get_restaurant_partners(
        "New York City",
        dietary_needs=["vegan"],
        limit=3
    )
    print(f"\nFound {len(vegan)} vegan-friendly restaurants:")
    for r in vegan:
        print(f"  - {r['name']}")
    
    print("✅ Restaurant partner tests passed!")


def test_lead_scoring():
    """Test lead qualification scoring."""
    print("\n=== Testing Lead Scoring ===")
    
    # Qualified lead
    score = calculate_lead_score(
        is_in_service_area=True,
        headcount=50,
        user_need="recurring"
    )
    print(f"Qualified lead score: {score['status']} ({score['score']} points)")
    print(f"Reasons: {', '.join(score['reasons'])}")
    assert score['status'] == "Qualified"
    
    # Maybe lead
    score = calculate_lead_score(
        is_in_service_area=True,
        headcount=15,
        user_need="one-time"
    )
    print(f"\nMaybe lead score: {score['status']} ({score['score']} points)")
    print(f"Reasons: {', '.join(score['reasons'])}")
    assert score['status'] == "Maybe"
    
    # Not qualified lead
    score = calculate_lead_score(
        is_in_service_area=False,
        headcount=5,
        user_need="one-time"
    )
    print(f"\nNot qualified lead score: {score['status']} ({score['score']} points)")
    print(f"Reasons: {', '.join(score['reasons'])}")
    assert score['status'] == "Not Qualified"
    
    print("✅ Lead scoring tests passed!")


def test_conversation_flow():
    """Test a full conversation flow through the agent."""
    print("\n=== Testing Conversation Flow ===")
    
    # Build the graph
    graph = build_graph()
    
    # Initialize state
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
    
    print("\n--- Stage 1: Greeting ---")
    result = graph.invoke(initial_state)
    last_message = result["messages"][-1].content
    print(f"Agent: {last_message[:150]}...")
    print(f"Stage: {result['conversation_stage']}")
    
    # Simulate user responding about their need
    print("\n--- Stage 2: User responds about recurring meals ---")
    result["messages"].append(HumanMessage(content="I need recurring daily lunches for my office"))
    result = graph.invoke(result)
    last_message = result["messages"][-1].content
    print(f"Agent: {last_message}")
    print(f"Stage: {result['conversation_stage']}")
    print(f"User need captured: {result.get('user_need')}")
    
    # Simulate location
    print("\n--- Stage 3: User provides location ---")
    result["messages"].append(HumanMessage(content="We're in New York City"))
    result = graph.invoke(result)
    last_message = result["messages"][-1].content
    print(f"Agent: {last_message}")
    print(f"Stage: {result['conversation_stage']}")
    print(f"Location: {result.get('location_city')}, {result.get('location_state')}")
    print(f"In service area: {result.get('is_in_service_area')}")
    
    print("\n✅ Conversation flow test completed!")
    print(f"Final stage reached: {result['conversation_stage']}")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("MEAL OUTPOST AGENT - TEST SUITE")
    print("=" * 60)
    
    try:
        test_service_area_check()
        test_restaurant_partners()
        test_lead_scoring()
        test_conversation_flow()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nYour agent is ready to deploy!")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()