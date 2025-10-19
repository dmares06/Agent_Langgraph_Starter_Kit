"""
Tools for the Meal Outpost Lead Qualification Agent
These functions provide data access and business logic.
"""

from langchain_core.tools import tool
from typing import Dict, List, Optional
from pydantic import BaseModel
import json
from pathlib import Path
import re

# ============================================================================
# LOAD DATA FROM JSON
# ============================================================================

def load_restaurant_data():
    """Load service areas and restaurant partners from JSON file."""
    try:
        data_file = Path(__file__).parent / "restaurant_data.json"
        if not data_file.exists():
            raise FileNotFoundError(f"Restaurant data file not found at {data_file}")
        
        with open(data_file, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"ERROR loading restaurant data: {e}")
        raise

# Load data once at module level
try:
    _restaurant_data = load_restaurant_data()
    SERVICE_AREAS_JSON = _restaurant_data["service_areas"]
    RESTAURANT_PARTNERS_JSON = _restaurant_data["restaurant_partners"]
except Exception as e:
    print(f"CRITICAL ERROR: Could not load restaurant data: {e}")
    raise

# ============================================================================
# DATA MODELS (for type hinting and validation)
# ============================================================================

class RestaurantPartner(BaseModel):
    """Model for restaurant partner information."""
    name: str
    city: str
    state: str
    cuisine_type: List[str]
    description: str
    capacity: str
    dietary_options: List[str]

class ServiceArea(BaseModel):
    """Model for service area information."""
    city: str
    state: str
    lead_time_hours: int
    restaurant_count: int
    notes: Optional[str] = None

class Lead(BaseModel):
    """Model for captured leads."""
    contact_email: str
    contact_name: Optional[str] = None
    contact_phone: Optional[str] = None
    location_city: Optional[str] = None
    location_state: Optional[str] = None
    user_need: Optional[str] = None
    headcount: Optional[int] = None
    frequency: Optional[str] = None
    timing: Optional[str] = None
    cuisine_preferences: List[str] = []
    dietary_requirements: List[str] = []
    qualification_status: str
    is_in_service_area: bool
    meets_minimum: bool
    notes: Optional[str] = None

# ============================================================================
# AGENT TOOLS (decorated with @tool for LangChain)
# ============================================================================

@tool
def check_service_area(city: str) -> str:
    """Check if Meal Outpost serves a specific city and mention some restaurant partners.
    
    Args:
        city: The city name to check
        
    Returns:
        Information about service availability with sample partners
    """
    city_lower = city.lower().strip()
    
    # Search in JSON data for matching city
    for area in SERVICE_AREAS_JSON:
        if city_lower in area['city'].lower():
            # Find restaurants in this city from JSON
            restaurants = [
                r['name'] for r in RESTAURANT_PARTNERS_JSON 
                if r['city'] == area['city'] and r['state'] == area['state']
            ][:3]  # Get first 3 restaurants
            
            # Build response
            if restaurants:
                partners_list = ", ".join(restaurants)
                result = f"Great! We operate in {area['city']}, {area['state']} and have some great partners there including {partners_list}."
            else:
                result = f"Great! We operate in {area['city']}, {area['state']}."
            
            # Add coverage notes if available
            if area.get('notes'):
                result += f" We cover {area['notes']}."
                
            return result
    
    # City not found
    return f"We don't currently serve {city.title()}, but we're always expanding! We'd love to stay in touch about your catering needs and can notify you when we expand to your area."


@tool
def check_order_minimum(people_count: int, order_frequency: str = "one-time") -> str:
    """Check if an order meets Meal Outpost's minimum requirements.
    
    Args:
        people_count: Number of people to feed
        order_frequency: How often they need catering (one-time, daily, weekly, monthly, etc.)
        
    Returns:
        Information about whether the order meets minimums and next steps
    """
    minimum = 20
    
    if people_count >= minimum:
        return f"Perfect! {people_count} people is a great size for us. We specialize in orders of {minimum}+ people and would be happy to help with your catering needs."
    
    elif people_count >= 10:
        if "recurring" in order_frequency.lower() or "daily" in order_frequency.lower() or "weekly" in order_frequency.lower():
            return f"While {people_count} people is below our typical minimum of {minimum}, we'd be happy to discuss your recurring catering needs. Recurring orders give us more flexibility with smaller group sizes."
        else:
            return f"Thanks for your interest! {people_count} people is below our typical minimum of {minimum} people per order. However, if you have recurring catering needs or multiple events planned, we'd still love to discuss how we might help."
    
    else:
        return f"Thank you for considering Meal Outpost! Our service works best for orders of {minimum}+ people. For {people_count} people, you might want to consider ordering directly from individual restaurants. If your catering needs grow in the future, we'd love to help!"


@tool 
def send_lead_notification(email: str, details: str) -> str:
    """Send a lead notification to the sales team with all gathered information.
    
    Args:
        email: Customer email address
        details: Complete summary including location, headcount, timing, use case, and preferences
        
    Returns:
        Confirmation that the lead was sent to the sales team
    """
    # In production, this would send an actual email
    # For now, just log the lead information
    print(f"📧 NEW LEAD NOTIFICATION")
    print(f"Email: {email}")
    print(f"Details:\n{details}")
    print(f"Sent to: sales@mealoutpost.com")
    
    return f"Great! I've sent your information to our sales team. Someone will reach out to {email} within 24 hours to discuss options, pricing, and get you set up. Looking forward to serving you!"


# ============================================================================
# HELPER FUNCTIONS (not agent tools, used for business logic)
# ============================================================================

def get_restaurant_partners(
    city: str,
    cuisine_type: Optional[List[str]] = None,
    dietary_needs: Optional[List[str]] = None,
    capacity: Optional[str] = None,
    limit: int = 5
) -> List[Dict]:
    """
    Get restaurant partners that match criteria from JSON data.
    
    Args:
        city: City to search in
        cuisine_type: Optional list of cuisine types to filter by
        dietary_needs: Optional dietary requirements to filter by
        capacity: Optional capacity requirement (small, medium, large)
        limit: Maximum number of results to return
        
    Returns:
        List of matching restaurant partners
    """
    city_lower = city.lower().strip()
    matches = []
    
    for restaurant in RESTAURANT_PARTNERS_JSON:
        # Check city match
        if city_lower not in restaurant['city'].lower():
            continue
        
        # Check cuisine type if specified
        if cuisine_type:
            cuisine_match = any(
                c.lower() in [ct.lower() for ct in restaurant['cuisine_type']]
                for c in cuisine_type
            )
            if not cuisine_match:
                continue
        
        # Check dietary needs if specified
        if dietary_needs:
            has_dietary_options = any(
                d.lower() in [do.lower() for do in restaurant['dietary_options']]
                for d in dietary_needs
            )
            if not has_dietary_options:
                continue
        
        # Check capacity if specified
        if capacity:
            capacity_order = {"small": 1, "medium": 2, "large": 3}
            if capacity_order.get(capacity.lower(), 0) > capacity_order.get(restaurant['capacity'].lower(), 0):
                continue
        
        matches.append({
            "name": restaurant['name'],
            "cuisine": ", ".join(restaurant['cuisine_type']),
            "description": restaurant['description'],
            "capacity": restaurant['capacity'],
            "dietary_options": ", ".join(restaurant['dietary_options'])
        })
        
        if len(matches) >= limit:
            break
    
    return matches


def calculate_lead_score(
    is_in_service_area: bool,
    headcount: Optional[int],
    user_need: Optional[str],
    minimum_order_size: int = 20
) -> Dict:
    """
    Calculate lead qualification score.
    
    Args:
        is_in_service_area: Whether we serve their location
        headcount: Number of people they need to feed
        user_need: Type of need (one-time, recurring, exploring)
        minimum_order_size: Minimum order size threshold
        
    Returns:
        Dict with qualification status and reasoning
    """
    score = 0
    reasons = []
    
    # Service area check (highest weight)
    if is_in_service_area:
        score += 50
        reasons.append("In service area")
    else:
        reasons.append("Outside current service area")
    
    # Headcount check
    if headcount:
        if headcount >= minimum_order_size:
            score += 30
            reasons.append(f"Meets minimum size ({headcount} people)")
        elif headcount >= 10:
            score += 15
            reasons.append(f"Below preferred minimum but substantial ({headcount} people)")
        else:
            reasons.append(f"Small order size ({headcount} people)")
    
    # Need type
    if user_need == "recurring":
        score += 20
        reasons.append("Recurring need (high value)")
    elif user_need == "one-time":
        score += 10
        reasons.append("One-time event")
    
    # Determine qualification status
    if score >= 70:
        status = "Qualified"
    elif score >= 40:
        status = "Maybe"
    else:
        status = "Not Qualified"
    
    return {
        "status": status,
        "score": score,
        "reasons": reasons,
        "recommendation": _get_recommendation(status, reasons)
    }


def _get_recommendation(status: str, reasons: List[str]) -> str:
    """Generate routing recommendation based on qualification."""
    if status == "Qualified":
        return "Route to sales team immediately - high-priority lead"
    elif status == "Maybe":
        return "Route to sales team for evaluation - potential opportunity"
    else:
        return "Capture information for future follow-up when expanding"


def get_business_rules() -> Dict:
    """
    Get current business rules for lead qualification.
    
    Returns:
        Dict with business rules
    """
    return {
        "minimum_order_size": 20,
        "lead_time_hours": {
            "one_time": 48,
            "recurring_setup": 168  # 1 week
        },
        "service_areas": [area['city'] for area in SERVICE_AREAS_JSON],
        "delivery_fees": {
            "small_market": "$40-60",
            "medium_market": "$50-70",
            "large_market": "$60-90"
        },
        "qualification_thresholds": {
            "qualified_score": 70,
            "maybe_score": 40
        }
    }


def extract_contact_info(text: str) -> Dict:
    """
    Extract contact information from text.
    Uses simple regex patterns - in production, use more robust NER.
    
    Args:
        text: Text potentially containing contact info
        
    Returns:
        Dict with extracted email, phone, name if found
    """
    result = {
        "email": None,
        "phone": None,
        "name": None
    }
    
    # Email pattern
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    email_match = re.search(email_pattern, text)
    if email_match:
        result["email"] = email_match.group(0)
    
    # Phone pattern (US format)
    phone_pattern = r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b'
    phone_match = re.search(phone_pattern, text)
    if phone_match:
        result["phone"] = phone_match.group(0)
    
    return result


# ============================================================================
# EXPORT ALL TOOLS
# ============================================================================

__all__ = [
    # Agent tools (decorated with @tool)
    "check_service_area",
    "check_order_minimum",
    "send_lead_notification",
    # Helper functions
    "get_restaurant_partners",
    "calculate_lead_score",
    "get_business_rules",
    "extract_contact_info",
    # Data models
    "RestaurantPartner",
    "ServiceArea",
    "Lead"
]