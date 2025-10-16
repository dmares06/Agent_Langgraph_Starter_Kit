"""
Tools for the Meal Outpost Lead Qualification Agent
These functions provide data access and business logic.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel
import json

# ============================================================================
# DATA MODELS
# ============================================================================

class RestaurantPartner(BaseModel):
    """Model for restaurant partner information."""
    name: str
    city: str
    state: str
    cuisine_type: List[str]
    description: str
    capacity: str  # small, medium, large
    dietary_options: List[str]
    avg_price_per_person: str  # "$15-25", "$25-40", etc.


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
# DATA STORAGE (Replace with actual database in production)
# ============================================================================

SERVICE_AREAS = [
    ServiceArea(
        city="New York City",
        state="NY",
        lead_time_hours=48,
        restaurant_count=25,
        notes="Manhattan, Brooklyn, Queens - outer boroughs may have higher delivery fees"
    ),
    ServiceArea(
        city="Los Angeles",
        state="CA",
        lead_time_hours=48,
        restaurant_count=20,
        notes="Greater LA area including Santa Monica, Culver City, Downtown"
    ),
    ServiceArea(
        city="Chicago",
        state="IL",
        lead_time_hours=48,
        restaurant_count=18
    ),
    ServiceArea(
        city="San Francisco",
        state="CA",
        lead_time_hours=48,
        restaurant_count=22,
        notes="SF proper and nearby cities like Oakland, Berkeley"
    ),
    ServiceArea(
        city="Boston",
        state="MA",
        lead_time_hours=48,
        restaurant_count=15
    ),
    ServiceArea(
        city="Washington",
        state="DC",
        lead_time_hours=48,
        restaurant_count=16,
        notes="DC, Arlington, Alexandria metro area"
    ),
    ServiceArea(
        city="Seattle",
        state="WA",
        lead_time_hours=48,
        restaurant_count=14
    ),
    ServiceArea(
        city="Austin",
        state="TX",
        lead_time_hours=48,
        restaurant_count=12
    ),
    ServiceArea(
        city="Denver",
        state="CO",
        lead_time_hours=48,
        restaurant_count=10
    ),
    ServiceArea(
        city="Miami",
        state="FL",
        lead_time_hours=48,
        restaurant_count=13
    ),
]


RESTAURANT_PARTNERS = [
    # New York City
    RestaurantPartner(
        name="Ben's Fast Food",
        city="New York City",
        state="NY",
        cuisine_type=["Italian"],
        description="Healthy delicious bowls with a variety of options",
        capacity="large",
        dietary_options=["vegetarian", "gluten-free"],
        avg_price_per_person="$25-35"
    ),
    RestaurantPartner(
        name="Pokeworks",
        city="New York City",
        state="NY",
        cuisine_type=["Healthy", "Vegetarian"],
        description="Incredible poke bowls with a variety of options",
        capacity="medium",
        dietary_options=["vegetarian", "vegan", "gluten-free", "dairy-free"],
        avg_price_per_person="$18-28"
    ),
    RestaurantPartner(
        name="Starbird Chicken",
        city="New York City",
        state="NY",
        cuisine_type=["BBQ", "American"],
        description="Delicious chicken meals",
        capacity="large",
        dietary_options=["gluten-free"],
        avg_price_per_person="$22-32"
    ),
    
    # Los Angeles
    RestaurantPartner(
        name="Pokeworks",
        city="Los Angeles",
        state="CA",
        cuisine_type=["Seafood", "California Cuisine"],
        description="Incredible poke bowls with a variety of options",
        capacity="medium",
        dietary_options=["vegetarian", "gluten-free", "pescatarian"],
        avg_price_per_person="$28-38"
    ),
    RestaurantPartner(
        name="Starbird Chicken",
        city="Los Angeles",
        state="CA",
        cuisine_type=["Mexican", "Latin"],
        description="Delicious chicken meals",
        capacity="large",
        dietary_options=["vegetarian", "vegan", "gluten-free"],
        avg_price_per_person="$16-24"
    ),
    
    # San Francisco
    RestaurantPartner(
        name="Curry Up Now",
        city="San Francisco",
        state="CA",
        cuisine_type=["Healthy", "Indian Street Food"],
        description="Modern Indian street food with authentic flavors",
        capacity="medium",
        dietary_options=["vegetarian", "vegan", "gluten-free", "dairy-free"],
        avg_price_per_person="$19-27"
    ),
]


# ============================================================================
# TOOL FUNCTIONS
# ============================================================================

def check_service_area(city: str, state: Optional[str] = None) -> Dict:
    """
    Check if we serve a specific location.
    
    Args:
        city: The city name
        state: Optional state abbreviation
        
    Returns:
        Dict with service area details or None if not served
    """
    city_lower = city.lower().strip()
    
    for area in SERVICE_AREAS:
        if city_lower in area.city.lower():
            if state and state.upper() != area.state:
                continue
            
            return {
                "serves_area": True,
                "city": area.city,
                "state": area.state,
                "lead_time_hours": area.lead_time_hours,
                "restaurant_count": area.restaurant_count,
                "notes": area.notes
            }
    
    return {
        "serves_area": False,
        "message": f"We don't currently serve {city}, but we're always expanding!"
    }


def get_restaurant_partners(
    city: str,
    cuisine_type: Optional[List[str]] = None,
    dietary_needs: Optional[List[str]] = None,
    capacity: Optional[str] = None,
    limit: int = 5
) -> List[Dict]:
    """
    Get restaurant partners that match criteria.
    
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
    
    for restaurant in RESTAURANT_PARTNERS:
        # Check city match
        if city_lower not in restaurant.city.lower():
            continue
        
        # Check cuisine type if specified
        if cuisine_type:
            cuisine_match = any(
                c.lower() in [ct.lower() for ct in restaurant.cuisine_type]
                for c in cuisine_type
            )
            if not cuisine_match:
                continue
        
        # Check dietary needs if specified
        if dietary_needs:
            has_dietary_options = any(
                d.lower() in [do.lower() for do in restaurant.dietary_options]
                for d in dietary_needs
            )
            if not has_dietary_options:
                continue
        
        # Check capacity if specified
        if capacity:
            capacity_order = {"small": 1, "medium": 2, "large": 3}
            if capacity_order.get(capacity.lower(), 0) > capacity_order.get(restaurant.capacity.lower(), 0):
                continue
        
        matches.append({
            "name": restaurant.name,
            "cuisine": ", ".join(restaurant.cuisine_type),
            "description": restaurant.description,
            "capacity": restaurant.capacity,
            "dietary_options": ", ".join(restaurant.dietary_options),
            "avg_price": restaurant.avg_price_per_person
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



def send_notification(
    lead_id: str,
    lead_data: Dict,
    routing_type: str = "general_sales"
) -> Dict:
    """
    Send email notification to sales team about new lead.
    No database storage - just email notification.
    """
    
    # Your sales email from config
    from .config import config
    sales_email = config.sales_email  # "sales@mealoutpost.com"
    
    contact = lead_data.get("contact", {})
    requirements = lead_data.get("requirements", {})
    qualification = lead_data.get("qualification", {})
    
    # Format email message
    email_content = f"""
New Catering Lead from Meal Outpost Website:

CONTACT INFORMATION:
- Email: {contact.get('email')}

CATERING REQUIREMENTS:
- Location: {requirements.get('location', {}).get('city')}, {requirements.get('location', {}).get('state')}
- Need Type: {requirements.get('need_type')}
- Group Size: {requirements.get('headcount')} people
- Timing: {requirements.get('timing')}
- Frequency: {requirements.get('frequency', 'One-time')}
- Cuisine Preferences: {', '.join(requirements.get('cuisine_preferences', []))}

QUALIFICATION:
- Status: {qualification.get('status')}
- In Service Area: {qualification.get('in_service_area')}
- Meets Minimum Size: {qualification.get('meets_minimum')}

ACTION REQUIRED: Follow up within 24 hours
    """
    
    # In production, replace this with actual email sending
    # For now, just log what would be sent
    print(f"EMAIL TO: {sales_email}")
    print(f"SUBJECT: New Catering Lead - {contact.get('email')}")
    print(f"CONTENT:\n{email_content}")
    
    return {
        "success": True,
        "recipient": sales_email,
        "message": "Email notification sent to sales team",
        "no_data_stored": True
    }


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
        "service_areas": [area.city for area in SERVICE_AREAS],
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
    import re
    
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
    "check_service_area",
    "get_restaurant_partners",
    "calculate_lead_score",
    "send_notification",
    "get_business_rules",
    "extract_contact_info",
    "RestaurantPartner",
    "ServiceArea",
    "Lead"
]