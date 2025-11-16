from openai import OpenAI
import pandas as pd
import json

# Configure the client to connect to your local vLLM server
client = OpenAI(
    api_key="a", 
    base_url="http://0.0.0.0:8000/v1" 
)



model_name = "Qwen/Qwen2.5-1.5B-Instruct"
json_path = "./dataset/questions/TF.jsonl"
results_path = "./results/baseline-" + model_name.split('/')[-1] + ".json"

questions = pd.read_json(json_path, lines=True, orient="records")

def infer(type="TF"): # TF or MC
    answers = []
    for index, row in questions.iterrows():
        if type == "TF":
            # Create a chat completion request
            response = client.chat.completions.create(
                model=model_name,  # Replace with your model's name, e.g., "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"[citation:7]
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Answer the following question with either 'True' or 'False' only."},
                    {"role": "user", "content": "Answer the following question with True or False only: " + row["Question"]}
                ],
                temperature=0.0  # Set low temperature for deterministic, fact-based answers
            )
        else:
            # Create a chat completion request
            response = client.chat.completions.create(
                model=model_name,
                # Replace with your model's name, e.g., "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"[citation:7]
                messages=[
                    {"role": "system",
                     "content": "You are a helpful assistant. Answer the following question with one letter, A, B, C, or D."},
                    {"role": "user",
                     "content": "Answer the following question with True or False only: " + row["Question"]}
                ],
                temperature=0.0  # Set low temperature for deterministic, fact-based answers
            )
        generated_answer = response.choices[0].message.content
        answers.append({
            "index" : index,
            "question" : row["Question"],
            "answer" : generated_answer
        })

    with open(results_path, "a", encoding='utf-8') as f:
        json.dump(answers, f, indent=4) 

def evaluation():
    results = pd.read_json(results_path)
    assert len(questions) == len(results)
    sum = 0
    total = len(questions)
    for index, row in questions.iterrows():
        if row["Answer"].lower() == results.iloc[index]["answer"].lower():
            sum += 1
    return sum/total


print(evaluation())