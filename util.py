import openai

def get_openai_embeddings(text):
  response = openai.Embedding.create(
    input=text,
    model="text-embedding-ada-002"
  )
  return response['data'][0]['embedding']

def get_bad_ux_reason(conversation):
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": f"Please find the reason for bad user experience if there is one. Please call set_bad_ux_reason always. Thanks! Here is the conversation logs:\n\n {conversation}"}],
        functions=[
            {
                "name": "set_bad_ux_reason",
                "description": "Provide the exact and concise reason for bad user experience if it exists",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "The reason for bad user experience if it exists, otherwise do not define",
                        },
                    },
                },
            }
        ],
    )

    fn_args = json.loads(completion.choices[0].message.function_call.arguments)
    return fn_args['reason'] if 'reason' in fn_args else None
