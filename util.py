import openai
from tenacity import retry, wait_exponential

@retry(wait=wait_exponential(multiplier=1, min=2, max=60))
def get_openai_embeddings(api_key, text):
    response = openai.Embedding.create(
        api_key=api_key, input=text, model="text-embedding-ada-002"
    )
    return response["data"][0]["embedding"]


@retry(wait=wait_exponential(multiplier=1, min=2, max=60))
def get_conversation_answer(api_key, conversation, question):
    completion = openai.ChatCompletion.create(
        api_key=api_key,
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": """You are a highly intelligent assistant with a deep understanding of product management. Given the conversation, provide a detailed and concise answer to the user's question. If there is no explicit information within the conversation, then you must respond with 'N/A'. If you are not sure, then respond with 'N/A'. Your response should always be a phrase less than or equal to 10 words!

Example #1 No explicit information

question: what did the user not like about the ai?

conversation:
joe: hi how are you?
ai: I'm doing well. How are you?
joe: good!

answer:
N/A

Example #2 Concise answers (10 words maximum!)

question: why did the user have frustration?

conversation:
billy: hi how are you?
ai: Honestly joe, how can you not know I'm an AI without feelings. I can't believe I have to talk with you. You are a horrible user and customer.
billy: wow I'll never use this product again

answer:
combative language and unnecessary insults
""",
            },
            {
                "role": "user",
                "content": f"Here is the conversation logs:\n{conversation}",
            },
            {"role": "user", "content": question},
        ]
    )

    answer = completion['choices'][0]['message']['content']
    return None if answer.lower() == 'n/a' else answer
