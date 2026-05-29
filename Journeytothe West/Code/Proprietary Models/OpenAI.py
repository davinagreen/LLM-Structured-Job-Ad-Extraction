from openai import OpenAI
import csv
from tqdm import tqdm
import ast
import chardet
import pandas as pd

"""
This program uses the OpenAI API to call GPT-series large language models for generating responses. 
Before running, please fill in your OpenAI API Key and select the correct model name.
"""

# Fill in OpenAI's API Key and select the model name (using GPT-4.1 as an example here)
api_key = "Your openai API key"  # TODO
model_name = "gpt-4.1"


# Classify the work arragements

# Set up the structured outputs format
functions_arragements = [
    {
        "name": "classify_arragements",
        "description": "Determine the remote work option for this position.",
        "parameters": {
            "type": "object",
            "properties": {
                "explanation": {
                    "type": "string",
                    "description": 'Your reasoning here based on the job description'
                },
                "Remote_option": {
                    "type": "string",
                    "enum": ['onsite', 'hybrid', 'remote']
                }
            },
            "required": ["explanation", "Remote_option"]
        }
    }
]


# Call the large language model to generate a response.
def work_arragements(model_name, api_key, text):
    history = [{"role": "system", "content": "You are a job advertisement analyst working for a recruitment platform.\n\n"
                                             "Please carefully read the following job description and classify the remote work option for this position."
                                             "Choose one from the following three categories: onsite, hybrid or remote\n"
                                             "Additionally, please provide a brief explanation for your choice based on the information in the job description.\n\n"
                                             "Output format:\n"
                                             "{'explanation': 'Your reasoning here based on the job description', 'Remote_option': 'One of [onsite, hybrid, remote]'}"
                }]
    client = OpenAI(api_key=api_key)
    answer = client.chat.completions.create(
        model=model_name,
        messages=history + [{"role": "user", "content": text}],
        functions=functions_arragements,
        function_call={"name": "classify_arragements"},
        temperature=0.1  # Set a low temperature for stable and precise responses.
    )
    return answer.choices[0].message.function_call.arguments


def handle_arragements():
    input_file = "cleaned_work_arrangements_test_set.csv"
    output_file = "arragement_result.csv"

    # Check the file encoding format
    with open(input_file, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']
        print(f"Detected encoding: {encoding}")

    # read the input csv file
    with open(input_file, "r", encoding=encoding) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    new_rows = []
    for row in tqdm(rows):
        try:
            id_ = row[0]
            job_ad = row[1]
            y_ture = row[2]

            # Iterate through each category of data and generate responses
            text = work_arragements(model_name, api_key, job_ad)
            text = ast.literal_eval(text)
            text = text["Remote_option"]

            # Determine if the LLM's response is correct
            is_same = str(y_ture == text)

            new_rows.append([id_, job_ad, y_ture, text, is_same])

        except Exception as e:
            print(f"An error occurred while processing the data: {row[0]} - error: {e}")
            continue

    # Write the results to a new csv file
    with open(output_file, "w", newline='', encoding=encoding) as f:
        writer = csv.writer(f)
        writer.writerow(header + ["LLM_answer", "Accuracy"])
        writer.writerows(new_rows)


# Classify the seniority

# Store all seniority types
seniority_categories = ['intermediate', 'senior', 'lead', 'head', 'experienced', 'entry level',
                        'executive', 'assistant', 'senior/lead', 'deputy', 'director', 'trainee',
                        'associate', 'graduate', 'junior', 'general-manager', 'coordinator', 'student',
                        'chief', 'principal', 'apprentice', 'qualified',
                        'entry-level to intermediate', 'senior associate', 'standard',
                        'senior assistant', 'specialist', 'mid-level', 'entry level assistant',
                        'experienced assistant', 'manager', 'graduate/junior', 'independent',
                        'senior lead', 'apprentice-first-year', '1st year apprentice',
                        'senior-executive', 'junior assistant', 'assistant manager', 'entry-level',
                        'supervisor', 'second-in-command', 'associate director', 'board',
                        '4th year apprentice', 'mid-senior', 'regional head', 'middle-management',
                        'advanced', '2nd year apprentice', 'intermediate apprentice', 'level 2',
                        'assistant head', 'owner', 'postdoctoral', 'owner-operator',
                        'middle management', 'senior head', 'assistant director',
                        'junior-intermediate', 'sous', 'post-doctoral', 'intermediate to senior', 'senior executive']

# Set up the structured outputs format
functions_seniority = [
    {
        "name": "classify_seniority",
        "description": "Classify the seniority level of a job based on its description.",
        "parameters": {
            "type": "object",
            "properties": {
                "explanation": {
                    "type": "string",
                    "description": 'Your reasoning here based on the job description'
                },
                "Seniority": {
                    "type": "string",
                    "enum": seniority_categories
                }
            },
            "required": ["explanation", "Seniority"]
        }
    }
]


# Call the large language model to generate a response.
def seniority(model_name, api_key, title, summary, details):
    history = [{"role": "system", "content": "You are a job advertisement analyst working for a recruitment platform.\n\n"
                                             "Please carefully read the following job description and classify the seniority level of a job based on its description."
                                             f"Choose one from the seniority categories list\n"
                                             "Additionally, please provide a brief explanation for your choice based on the information in the job description.\n\n"
                                             "Output format:\n"
                                             "{'explanation': 'Your reasoning here based on the job description', 'Seniority': 'one of the seniority categories list'}"
                }]
    client = OpenAI(api_key=api_key)
    answer = client.chat.completions.create(
        model=model_name,
        messages=history + [{"role": "user", "content": f"job title:{title} job summary:{summary} job details:{details}"}],
        functions=functions_seniority,
        function_call={"name": "classify_seniority"},
        temperature=0.1  # Set a low temperature for stable and precise responses.
    )
    return answer.choices[0].message.function_call.arguments


# Group equivalent seniority types with identical meanings.
def normalize_label(label):
    if not label:
        return ""

    label = label.strip().lower()

    if label in ["entry-level"]:
        return "entry level"
    if label in ["post-doctoral"]:
        return "postdoctoral"
    if label in ["middle management"]:
        return "middle-management"
    if label in ["apprentice-first-year"]:
        return "1st year apprentice"

    return label


def handle_seniority():
    input_file = "cleaned_seniority_labelled_test_set.csv"
    output_file = "seniority_result.csv"

    # Check the file encoding format
    with open(input_file, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']

    # read the input csv file
    with open(input_file, "r", encoding=encoding) as f:
        reader = csv.reader(f)
        next(reader)
        rows = list(reader)

    new_rows = []
    for row in tqdm(rows):
        try:
            id_ = row[0]
            job_title = row[1]
            job_summary = row[2]
            job_details = row[3]
            y_ture = row[6]

            # Iterate through each category of data and generate responses
            text = seniority(model_name, api_key, job_title, job_summary, job_details)
            text = ast.literal_eval(text)
            text = text["Seniority"]

            # Determine if the LLM's response is correct
            text = normalize_label(text)
            is_same = str(y_ture == text)

            new_rows.append([id_, job_title, job_summary, job_details, y_ture, text, is_same])

        except Exception as e:
            print(f"An error occurred while processing the data: {row[0]} - error: {e}")
            continue

    # Write the results to a new csv file
    with open(output_file, "w", newline='', encoding=encoding) as f:
        writer = csv.writer(f)
        writer.writerow(["job_id", "job_title", "job_summary", "job_ad_details", "y_true", "LLM_answer", "Accuracy"])
        writer.writerows(new_rows)


# Classify the salary

# Set up the structured outputs format
functions_salary = [
    {
        "name": "determine_salary",
        "description": "Determine the salary of a job based on its description.",
        "parameters": {
            "type": "object",
            "properties": {
                "explanation": {
                    "type": "string",
                    "description": 'Your reasoning here based on the job description'
                },
                "Salary": {
                    "type": "string",
                    "description": "[salary_lower_bound]-[salary_upper_bound]-[currency_code]-[time_unit]"
                }
            },
            "required": ["explanation", "salary"]
        }
    }
]


# Call the large language model to generate a response.
def salary(model_name, api_key, title, details, nation):
    history = [{"role": "system", "content": "You are a job advertisement analyst working for a recruitment platform.\n\n"
                                             "Please carefully read the following job description and determine the salary of a job based on its description."
                                             "Extract the salary in the following strict format,and strictly follow the rules below:[salary_lower_bound]-[salary_upper_bound]-[currency_code]-[time_unit]\n"
                                             "For example: 1500-1800-myr-monthly.\n"
                                             "currency_code must be the lowercase version of ISO 4217 three-letter currency code (e.g., myr, usd, hkd). "
                                             "time_unit must be one of: hourly, daily, weekly, monthly, or annual.\n"
                                             "If the job description does not mention any salary, return this exact value: 0-0-none-none\n\n"
                                             "Additionally, please provide a brief explanation for your choice based on the information in the job description.\n\n"
                                             "Output format:\n"
                                             "{'explanation': 'Your reasoning here based on the job description', 'Salary': '[salary_lower_bound]-[salary_upper_bound]-[currency_code]-[time_unit]'}"
                }]
    client = OpenAI(api_key=api_key)
    answer = client.chat.completions.create(
        model=model_name,
        messages=history + [{"role": "user", "content": f"job title:{title} job details:{details} Country where the job is located:{nation}"}],
        functions=functions_salary,
        function_call={"name": "determine_salary"},
        temperature=0.1  # Set a low temperature for stable and precise responses.
    )
    return answer.choices[0].message.function_call.arguments


def handle_salary():
    input_file = "cleaned_salary_labelled_test_set.csv"
    output_file = "salary_result.csv"

    # read the input csv file
    df = pd.read_csv(input_file)
    rows = df.values.tolist()
    header = df.columns.tolist()

    new_rows = []
    for row in tqdm(rows):
        try:
            id_ = row[0]
            job_title = row[1]
            job_details = row[2]
            nation = row[3]
            y_ture = row[5]

            # Iterate through each category of data and generate responses
            text = salary(model_name, api_key, job_title, job_details, nation)
            text = ast.literal_eval(text)
            text = text["Salary"]

            # Determine if the LLM's response is correct
            is_same = str(y_ture == text)

            new_rows.append([id_, job_title, job_details, nation, y_ture, text, is_same])

        except Exception as e:
            print(f"An error occurred while processing the data: {row[0]} - error: {e}")
            continue

    # Write the results to a new csv file
    with open(output_file, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["job_id", "job_title", "job_ad_details", "nation_short_desc", "y_true", "LLM_answer", "Accuracy"])
        writer.writerows(new_rows)


# Classify the keyword

# Store all keyword types
keyword_categories = ['beauty', 'office worker', 'blue-collar worker', 'nurse', 'food&restraurant ', 'manager',
                      'engineer', 'real estate', 'accounting', 'driver', 'recruiter ', 'analyst', 'retail', 'legal',
                      'medical ', 'chef', 'electritian', 'marketing', 'blue-collar worker ', 'researcher', 'delivery ',
                      'architect', 'defence', 'scientist', 'education', 'housekeeper', 'Horticulturist', 'recruiter']

# Set up the structured outputs format
functions_keyword = [
    {
        "name": "classify_keyword",
        "description": "Classify the keyword of a job based on its description.",
        "parameters": {
            "type": "object",
            "properties": {
                "explanation": {
                    "type": "string",
                    "description": 'Your reasoning here based on the job description'
                },
                "Keyword": {
                    "type": "string",
                    "enum": keyword_categories
                }
            },
            "required": ["explanation", "Keyword"]
        }
    }
]


# Call the large language model to generate a response.
def keyword(model_name, api_key, title, details):
    history = [{"role": "system", "content": "You are a job advertisement analyst working for a recruitment platform.\n\n"
                                             "Please carefully read the following job description and classify the keyword of a job based on its description."
                                             f"Choose one from the keyword categories list\n"
                                             "Additionally, please provide a brief explanation for your choice based on the information in the job description.\n\n"
                                             "Output format:\n"
                                             "{'explanation': 'Your reasoning here based on the job description', 'Keyword': 'one of the keyword categories list'}"
                }]
    client = OpenAI(api_key=api_key)
    answer = client.chat.completions.create(
        model=model_name,
        messages=history + [{"role": "user", "content": f"job title:{title} job details:{details}"}],
        functions=functions_keyword,
        function_call={"name": "classify_keyword"},
        temperature=0.1  # Set a low temperature for stable and precise responses.
    )
    return answer.choices[0].message.function_call.arguments


def handle_keyword():
    input_file = "job_keyword_test_set.csv"
    output_file = "keyword_result.csv"

    # Check the file encoding format
    with open(input_file, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']

    # read the input csv file
    df = pd.read_csv(input_file, encoding=encoding)
    rows = df.values.tolist()
    header = df.columns.tolist()

    new_rows = []
    for row in tqdm(rows):
        try:
            job_title = row[0]
            job_details = row[1]
            job_keyword_true = row[3]

            # Iterate through each category of data and generate responses
            text = keyword(model_name, api_key, job_title, job_details)
            text = ast.literal_eval(text)
            text = text["Keyword"]

            # Determine if the LLM's response is correct
            is_same = str(job_keyword_true == text)

            new_rows.append([job_title, job_details, job_keyword_true, text, is_same])

        except Exception as e:
            print(f"An error occurred while processing the data: {row[0]} - error: {e}")
            continue

    # Write the results to a new csv file
    with open(output_file, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["job_title", "job_details", "job_keyword_true", "LLM_answer", "Accuracy"])
        writer.writerows(new_rows)


# Test the LLM's performance on the following four question: work_arragements, seniority, salary, keyword
handle_arragements()
handle_seniority()
handle_salary()
handle_keyword()
