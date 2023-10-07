import argparse

def main():
    parser = argparse.ArgumentParser(description="My CLI Tool")
    parser.add_argument('--csv', type=str, required=True, help='Path to conversation logs in csv format')
    parser.add_argument('--question', type=str, required=True, help='Question to be asked for each conversation/session')

    args = parser.parse_args()

    print(f'CSV file path: {args.csv}')
    print(f'Question: {args.question}')

if __name__ == '__main__':
    main()
