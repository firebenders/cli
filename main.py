import argparse
import os
from dotenv import load_dotenv
import csv
from util import get_conversation_answer, get_openai_embeddings
from tqdm import tqdm

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run quick analytics on a set of unstructured conversation data.")
    parser.add_argument('--csv', type=str, required=True, help='Path to conversation logs in csv format')
    parser.add_argument('--question', type=str, required=True, help='Question to be asked for each conversation/session')

    args = parser.parse_args()

    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Load CSV file and check format
    with open(args.csv, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)[:5]
    for row in data:
        if not isinstance(row[0], str):
            raise ValueError("CSV file format is incorrect. Each row should have one column that is a string.")

    print(f'CSV file path: {args.csv}')
    print(f'Question: {args.question}')

    # Create a new data structure for conversations
    conversations = []
    for i, row in tqdm(enumerate(data), total=len(data), desc="Processing conversations"):
        conversation = row[0]
        gptAnswer = get_conversation_answer(openai_api_key, conversation, args.question)
        gptEmbedding = get_openai_embeddings(openai_api_key, gptAnswer) if gptAnswer else None
        conversations.append({
            'id': i,
            'conversation': conversation,
            'gptAnswer': gptAnswer,
            'gptEmbedding': gptEmbedding,
        })
    
    # Print conversations in a more readable format
    for conversation in conversations:
        print(f"ID: {conversation['id']}")
        print(f"Conversation: {conversation['conversation']}")
        print(f"GPT Answer: {conversation['gptAnswer']}")
        print("\n")

if __name__ == '__main__':
    main()
