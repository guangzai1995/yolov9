import requests
import json
import os

# 假设这是你的API端点，需要根据实际情况修改
API_URL = 'http://0.0.0.0:8000/v1/score'
API_KEY = '123456'

prefix = """<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
"""
suffix = """<|im_end|>
<|im_start|>assistant
<think>

</think>

"""

query_template = """{prefix}<Instruct>: {instruction}
<Query>: {query}"""
document_template = "<Document>: {doc}{suffix}"


def main() -> None:
    instruction = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )

    queries = [
        "What is the capital of China?",
        "Explain gravity",
    ]

    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ]

    queries = [
        query_template.format(prefix=prefix, instruction=instruction, query=query)
        for query in queries
    ]
    documents = [document_template.format(doc=doc, suffix=suffix) for doc in documents]

    headers = {
        'Content-Type': 'application/json',
    }
    if API_KEY:
        headers['Authorization'] = f'Bearer {API_KEY}'

    data = {
        "text_1": queries,
        "text_2": documents
    }

    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        scores = response.json()
        print("-" * 30)
        print(scores)
        print("-" * 30)
    except requests.exceptions.RequestException as e:
        print(f"请求出错: {e}")


if __name__ == "__main__":
    main()