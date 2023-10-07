import argparse
import os
from dotenv import load_dotenv

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run quick analytics on a set of unstructured conversation data.")
    parser.add_argument('--csv', type=str, required=True, help='Path to conversation logs in csv format')
    parser.add_argument('--question', type=str, required=True, help='Question to be asked for each conversation/session')

    args = parser.parse_args()

    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    print(f'CSV file path: {args.csv}')
    print(f'Question: {args.question}')

if __name__ == '__main__':
    main()
