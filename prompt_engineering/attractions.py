
who_is_traveling = "family"
num_adults = 2
num_kids = 0
num_elderly = 0
num_friends = 0
num_kids_friends = 0
destination = "Yosemite National Park"
num_attractions = 5
num_days = 2

list_preferences = [
    "easy and kid friendly hikes",
    "interest in nature photography"
]

starting_point = "Yosemite National Park Visitor Center"
month_travel = "February"

list_properties = [
    ("num_miles", "number of miles away from the starting point"),
    ("how_to_reach", "explain some guidelines or landmarks or instructions to easily find the attraction"),
    ("water_station", "what are the solutions if they run out of drinking water"),
    ("food_recommendation", "what are the food recommendations, restaurants, and should they carry food"),
    ("public_restrooms", "how far is the closest public restroom"),
    ("weather_conditions", "what to expect regarding weather and how can they prepare well"),
    ("is_drivable", "is the attraction point drivable, how much walking is expected, is it accessible")
]

output_format = "YAML"

prompt = f"""
Acting as a thorough travel planner, consider the following points:
A journey plan has to be created for a family of {num_adults} adults,
{num_kids} kids, {num_elderly} elderly people.
Their destination is {destination}. 
They would like to visit {num_attractions} attractions at the destination spread across {num_days} days.
They have the following {len(list_preferences)} preferences: {list_preferences}.

The output is expected in the {output_format} format. The list of properties contains {len(list_properties)}
is in a tuple format (property_name, query): {list_properties}
Each attraction must contain some valid response for each of the properties and you may do web search if required.

Create a detailed list of {num_attractions} attractions in the month of {month_travel} at {destination} assuming they start at {starting_point} with the above properties and instructions and output in {output_format}:
"""

# need to clean up
ch_nnbsp = " "

print(prompt)