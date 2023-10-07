import json
import openai

def get_openai_embeddings(text):
  response = openai.Embedding.create(
    input=text,
    model="text-embedding-ada-002"
  )
  return response['data'][0]['embedding']

def get_conversation_answer(conversation, question):
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
           {"role": "system", "content": "You are a helpful product manager. Given the conversation, answer the user's question as concisely as possible. Please always use the defined functions to provide an answer!"},
           {"role": "user", "content": f"Here is the conversation logs:\n\n {conversation}"},
           {"role": "user", "content": question}
           ],
        functions=[
            {
                "name": "answer",
                "description": "Provide concise answer to user request, and if there is no clear answer then do not specify it",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "The answer",
                        },
                    },
                },
            }
        ],
    )

    fn_args = json.loads(completion.choices[0].message.function_call.arguments)
    return fn_args['answer'] if 'answer' in fn_args else None
