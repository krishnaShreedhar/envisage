import yaml

from attractions import list_properties

with open("../data/yosemite_np_20250614.yaml", "r") as fh:
    dict_yaml_data = yaml.safe_load(fh)

sample_info = dict_yaml_data["days"][0]["attractions"][0]
# print(sample_info)

list_keys = [prop[0] for prop in list_properties]

prefix = f"""
Act as an expert Data Engineer, and ensure that the given input has some value 
for all the keys in the list {list_keys}.
If some values do not have, create a list of keys that have missing values. 
You may use code in python to validate the input. 
"""

action = f"""
Validate the input based on the guidelines.
"""

prompt = f"""
guideline: {prefix}

input: {sample_info}

action: {action}
"""

if __name__ == '__main__':
    print(prompt)
