import os
import math
from typing import Optional, Tuple
from huggingface_hub import login
from dotenv import load_dotenv
from PIL import Image
from smolagents import CodeAgent,InferenceClientModel,tool,DuckDuckGoSearchTool,VisitWebpageTool,TransformersModel
import requests
from smolagents.models import ChatMessage
from huggingface_hub import InferenceClient


client = InferenceClient()

load_dotenv()

login(token = os.getenv("HF_TOKEN"), add_to_git_credential = False)

@tool
def calculate_cargo_travel_time(origin_coords: Tuple[float, float], destination_coords: Tuple[float, float], cruising_speed_kmh: Optional[float] = 750.0) -> float:
    """
    Calculate the travel time for a cargo plane between two points on Earth using great-circle distance.
    
    Args:
        origin_coords (Tuple[float, float]): A tuple containing the latitude and longitude of the origin point in degrees.
        destination_coords (Tuple[float, float]): A tuple containing the latitude and longitude of the destination point in degrees.
        cruising_speed_kmh (Optional[float]): The cruising speed of the cargo plane in kilometers per hour. Default is 750 km/h.

    Returns:
        float: The estimated travel time in hours.

    Example:
        >>> # Chicago (41.8781° N, 87.6298° W) to Sydney (33.8688° S, 151.2093° E)
        >>> result = calculate_cargo_travel_time((41.8781, -87.6298), (-33.8688, 151.2093))
    """

    def to_radians(degrees: float) -> float:
        return degrees * math.pi / 180.0
    
    lat1, lon1 = map(to_radians, origin_coords)
    lat2, lon2 = map(to_radians, destination_coords)

    EARTH_RADIUS_KM = 6371.0

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance_km = EARTH_RADIUS_KM * c
    
    actual_distance_km = distance_km * 1.1

    flight_time = (actual_distance_km / cruising_speed_kmh) + 1.0

    return round(flight_time, 2)

print(calculate_cargo_travel_time((41.8781, -87.6298), (-33.8688, 151.2093)))

agent = CodeAgent(
    model=InferenceClientModel(),
    tools=[calculate_cargo_travel_time, DuckDuckGoSearchTool(), VisitWebpageTool()],
    additional_authorized_imports = ['pandas'],
    max_steps = 20
    )

task = """Find all Batman filming locations in the world, calculate the time to transfer via cargo plane to here (we're in Gotham, 40.7128° N, 74.0060° W), and return them to me as a pandas dataframe.
Also give me some supercar factories with the same cargo plane transfer time."""

agent.planning_interval = 4

detailed_report = agent.run(f"""
You're an expert analyst. You make comprehensive reports after visiting many websites.
Don't hesitate to search for many queries at once in a for loop.
For each data point that you find, visit the source url to confirm numbers.

{task}
""")

print(detailed_report)
