"""
Software Requirements Specification for the Web Application

"""
import yaml

sr_prefix = """
Act as a Lead Front-end and Backend Software Developer and generate code for the following
Software Requirements Specification:
"""

sr_app = f"""
Web App requirements:
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
1. Create two dropdowns:
    1. List of travel destinations 
    2. List of months
2. And, create a tab-based details section that will populate values based on the values selected 
   from the above two dropdowns
"""

p_form = """
The form page that is able to send en email after gathering the following input data:
1. Destination name -- string only -- maximum of 200 characters
2. List of preferences -- string only -- maximum of 200 characters
3. Number of days -- positive integer only
4. Month of travel -- dropdown list of 12 calendar months
5. Additional travel requirements section -- string only -- maximum of 1000 characters

Email should be sent to a configurable email address with a fixed title 'journey plan request'
"""

p_about = """
The about page:
1. Create a place holder for travel pictures and text in zig zag timeline pattern.
3. Write warm and inclusive travel quotes
"""

list_pages = [
    p_home,
    p_explorer,
    p_form,
    p_about
]

str_list_pages = "".join(list_pages)

sr_pages = f"""
Web pages requirements:
The pages must be easily navigable with a menu ribbon
The web app can have the following {len(list_pages)} pages:
{str_list_pages}
"""

sr_css_features = f"""
Web page CSS requirements:
The web app is for travel planning and thus should be minimalistic, warm, very welcoming and intuitive.
The color combination must be ambient utilizing sunset and sky based colors.
The buttons and filters on the website should have a smooth color transition from light when unselected and 
dark when hovered over, selected or clicked.
"""

file_path = "../data/yosemite_np_20250614.yaml"
with open(file_path, "r") as fh:
    sample_yaml_travel_plan = yaml.safe_load(fh)

sr_data_feed = f"""
Data feed requirements:
The data feed would include multiple YAML file based travel plans:
Sample travel plan YAML file looks as follows:
{sample_yaml_travel_plan}

Create python based data transformation to easily feed into the web app's client-side javascript 
for easy user interaction with the data, and to generate a metadata CSV file for managing future 
travel plan files.

Write a client-side Java-script to appropriately feed into the HTML explorer page to display content,
and interact with the content without modifications to the data.
"""

sr_interactivity = """
Web page interactivity requirements:
Following are the expected user journeys and interactions with the static serverless web-app:
1. Able to navigate the web pages easily
2. Read content of home and about pages
3. Navigate to the explorer page
    1. In the explorer page select options from the dropdown boxes
    2. When one dropdown value is changed, the list of values in the other dropdown must change 
       according to the available data feed.
    3. The tabs data must populate the day-wise plans from the data feed
4. Users must be able to fill out the forms so that the form data be sent as an email.
"""

sr_code_quality = """
Code quality requirements:
The code files must have the following features:
1. code must be easily readable and modular
2. Add code comments as much as possible
3. Handle exceptions or missing data
"""

sr_postfix_instruction = f"""
Create the web app based on the above instructions about web-app, 
the pages in the web app, HTML and CSS requirements,
python data manipulator, javascript based interactivity on the webpages,
ensure code quality requirements.
"""

list_requirements = [
    sr_prefix,
    sr_app,
    sr_pages,
    sr_css_features,
    sr_data_feed,
    sr_interactivity,
    sr_code_quality,
    sr_postfix_instruction
]

prompt = "".join(list_requirements)

print(prompt)