import argparse
import os
from dotenv import load_dotenv
import csv
from util import get_conversation_answer, get_openai_embeddings
from cluster import cluster, write_to_csv, InputData, Conversation
from tqdm import tqdm
import os
import pickle

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run quick analytics on a set of unstructured conversation data.")
    parser.add_argument('--csv', type=str, required=True, help='Path to conversation logs in csv format')
    parser.add_argument('--question', type=str, required=True, help='Question to be asked for each conversation/session')
    parser.add_argument('--output', type=str, required=True, help='Path to output csv file')

    args = parser.parse_args()

    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Load CSV file and check format
    with open(args.csv, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip the header
        data = list(reader)
    for row in data:
        if not isinstance(row[0], str):
            raise ValueError("CSV file format is incorrect. Each row should have one column that is a string.")

    print(f'CSV file path: {args.csv}')
    print(f'Question: {args.question}')

    # Create a new data structure for conversations
    conversations = []
    answer_cache = {}
    embedding_cache = {}
    try:
        with open('answer_cache.pkl', 'rb') as cache_file:
            answer_cache = pickle.load(cache_file)
        with open('embedding_cache.pkl', 'rb') as cache_file:
            embedding_cache = pickle.load(cache_file)
    except (FileNotFoundError, EOFError):
        pass
    try:
        for i, row in tqdm(enumerate(data), total=len(data), desc="Processing conversations"):
            conversation = row[0]
            answer_cache_key = f"{conversation}_{args.question}"
            if answer_cache_key not in answer_cache:
                gptAnswer = get_conversation_answer(openai_api_key, conversation, args.question)
                answer_cache[answer_cache_key] = gptAnswer
            else:
                gptAnswer = answer_cache[answer_cache_key]
            if gptAnswer not in embedding_cache:
                gptEmbedding = get_openai_embeddings(openai_api_key, gptAnswer) if gptAnswer else None
                embedding_cache[gptAnswer] = gptEmbedding
            else:
                gptEmbedding = embedding_cache[gptAnswer]
            conversation_obj = Conversation(id=i, conversation=conversation, gptAnswer=gptAnswer, gptEmbedding=gptEmbedding)
            conversations.append(conversation_obj)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        with open('answer_cache.pkl', 'wb') as cache_file:
            pickle.dump(answer_cache, cache_file)
        with open('embedding_cache.pkl', 'wb') as cache_file:
            pickle.dump(embedding_cache, cache_file)

    cluster_output = cluster(
        InputData(
            question=args.question,
            conversations=conversations
        )
    )
    write_to_csv(cluster_output, filename=args.output)


if __name__ == '__main__':
    main()
