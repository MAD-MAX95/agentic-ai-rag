import requests


BASE_URL = "http://127.0.0.1:8000/chat"


queries = [
    "What is Agentic AI according to the eBook?",
    "How do AI agents differ from traditional automation systems?",
    "What are the core components of an Agentic Architecture?",
    "What role does memory play in Agentic AI workflows?",
    "Who won the 2022 FIFA World Cup?",
]


for query in queries:

    print("\n" + "=" * 80)
    print("QUERY:")
    print(query)

    response = requests.post(
        BASE_URL,
        json={"query": query},
    )

    print("\nSTATUS:", response.status_code)

    if response.ok:
        data = response.json()

        print("\nANSWER:")
        print(data["final_answer"])

        print("\nCONFIDENCE:")
        print(data["confidence_score"])

    else:
        print(response.text)