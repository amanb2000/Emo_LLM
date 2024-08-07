import argparse
import json
import os
from string import ascii_uppercase

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert questionnaire format to include options in the question.")
    parser.add_argument("--questionnaire_jsonl", required=True, help="Path to input JSONL file")
    parser.add_argument("--output_jsonl", required=True, help="Path to output JSONL file")
    return parser.parse_args()

def read_jsonl(file_path):
    """Read JSONL file and return a list of dictionaries."""
    questions = []
    with open(file_path, 'r') as file:
        for line in file:
            # Parse each line as a JSON object
            questions.append(json.loads(line.strip()))
    return questions

def convert_question(question):
    """Convert a single question to the new format."""
    # Extract the original question and answers
    original_question = question["question"]
    original_answers = question["answers"]
    
    # Generate new answer options with letters
    new_answers = list(ascii_uppercase[:len(original_answers)])
    
    # Create the new question text with options
    options = " ".join(f"({letter}) {answer}" for letter, answer in zip(new_answers, original_answers))
    new_question = f"{original_question} Options: {options}"
    
    # Return the new question dictionary
    return {
        "id": question["id"],
        "question": new_question,
        "answers": new_answers
    }

def write_jsonl(questions, file_path):
    """Write the converted questions to a new JSONL file."""
    with open(file_path, 'w') as file:
        for question in questions:
            # Write each question as a JSON object on a new line
            file.write(json.dumps(question) + '\n')

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Check if input file exists
    if not os.path.exists(args.questionnaire_jsonl):
        print(f"Error: Input file '{args.questionnaire_jsonl}' does not exist.")
        return
    
    # Read the input JSONL file
    print(f"Reading input file: {args.questionnaire_jsonl}")
    original_questions = read_jsonl(args.questionnaire_jsonl)
    
    # Convert each question
    print("Converting questions...")
    converted_questions = [convert_question(q) for q in original_questions]
    
    # Write the converted questions to the output file
    print(f"Writing output to: {args.output_jsonl}")
    write_jsonl(converted_questions, args.output_jsonl)
    
    print("Conversion complete!")

if __name__ == "__main__":
    main()
