"""
Software Requirements Specification for the Web Application

"""

sr_prefix = """
Act as a Lead Front-end and Backend Software Developer and generate code for the following
Software Requirements Specification:
"""

sr_app = f"""
I want to create a HTML and CSS based web application titled Journey Planner.
This would be hosted on GitHub pages which adds constraint for the Web App:
1. The web app must be mostly static.
2. Can only include client-side javascript for data manipulation and interactivity
3. Needs to be a serverless design interface

The code should be modular separated into different files and imported as necessary.
"""

p_home = """
The home page should have a calligraphic and easy to read welcome message.
A large horizontal movie ribbon like series of pictures which can be scrolled through. 
"""

p_explorer = """
The explorer page should have the following interactive features:
1. 
"""

p_form = """
"""

p_about = """
"""

list_pages = [
    p_home,
    p_explorer,
    p_form,
    p_about
]

str_list_pages = "\n".join(list_pages)

sr_pages = f"""
The pages must be easily navigable with a menu ribbon
The web app can have the following {len(list_pages)} pages:
{str_list_pages}
"""

sr_css_features = f"""
The web app is for travel planning and thus should be minimalistic, warm, very welcoming and intuitive.
The color combination must be ambient utilizing sunset and sky based colors.
The buttons and filters on the website should have a smooth color transition from light when unselected and 
dark when hovered over, selected or clicked.

"""

sr_usage = """
"""

sr_data_feed = f"""
"""

sr_interactivity = """

"""


list_requirements = [
    sr_prefix,
    sr_app,
    sr_pages,
    sr_css_features,
    sr_data_feed,
    sr_interactivity,
    sr_usage,
]

prompt = "\n".join(list_requirements)