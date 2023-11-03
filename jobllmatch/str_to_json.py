import json
import logging
import re

with open("sample_job_description.txt", "r") as file:
    job_description_text = file.read()


# Define regular expressions to extsract information
regex_pattern = r"Job Description: (?P<job_title>.+?)(?=(?:\nJob Description:|$))"
matches = re.finditer(regex_pattern, job_description_text, re.DOTALL)

# Create a list to store the extracted information for each job description
job_info_list = []

for match in matches:
    job_info = match.group("job_title")
    
    # Define regular expressions to extract key-value pairs for each job description
    job_info_regex = r"(?P<key>[\w\s]+):\s(?P<value>.+)"
    job_info_matches = re.finditer(job_info_regex, job_info)
    
    # Create a dictionary to store the extracted information for each job description
    job_info_dict = {}
    for info_match in job_info_matches:
        key = info_match.group("key")
        value = info_match.group("value")
        job_info_dict[key] = value
    
    job_info_list.append(job_info_dict)

# Save the extracted information to a JSON file
with open("job_descriptions.json", "w") as json_file:
    json.dump(job_info_list, json_file, indent=4)

logging.info("Job descriptions information saved to 'job_descriptions.json'.")
