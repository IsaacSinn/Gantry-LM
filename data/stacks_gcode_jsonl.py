from datasets import load_dataset
import requests
import os
import time
from bs4 import BeautifulSoup
import argparse
import random
import openai
import json
from dotenv import load_dotenv

# Load environment variables from .env file (for API key)
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def download_gcode_content(repo_name, file_path):
    """Download G-code content from GitHub repository."""
    # Construct GitHub URL
    github_url = f"https://github.com/{repo_name}/blob/master/{file_path}"
    
    try:
        # Add delay to avoid rate limiting
        time.sleep(0.5)
        
        # Send request to GitHub
        response = requests.get(github_url)
        response.raise_for_status()
        
        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the raw content element (GitHub stores code in a specific element)
        code_element = soup.select_one('table.highlight')
        if code_element:
            # Extract text content from the code element
            gcode_content = code_element.get_text()
            return gcode_content
        else:
            # Try to get raw content directly
            raw_url = f"https://raw.githubusercontent.com/{repo_name}/master/{file_path}"
            raw_response = requests.get(raw_url)
            raw_response.raise_for_status()
            return raw_response.text
    except Exception as e:
        print(f"Error downloading {file_path} from {repo_name}: {e}")
        return None

def generate_user_command(repo_name, file_path, gcode_content):
    """Generate a plausible user command using OpenAI's ChatGPT API."""
    try:
        # Get the first 20 lines of G-code for context
        gcode_lines = gcode_content.split('\n')[:20]
        gcode_preview = '\n'.join(gcode_lines)
        
        # Create prompt for ChatGPT
        prompt = f"""
        Based on this GitHub repository name: "{repo_name}", G-code filename: "{file_path}", 
        and the first 20 lines of G-code content:

        ```
        {gcode_preview}
        ```
        
        Generate a single-sentence user command that might have produced this G-code file.
        
        The command should be something like "3D print a custom bracket" or 
        "Create a 2D engraving of a tree". Keep it concise and focused on what the G-code might create, but try to be specific.
        
        Return only the command text with no additional explanation or formatting.
        """
        
        # Call OpenAI API using the new client format
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates concise G-code descriptions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0.7
        )
        
        # Extract and clean the response
        user_command = response.choices[0].message.content.strip()
        
        # Remove quotes if present
        user_command = user_command.strip('"\'')
        
        return user_command
    
    except Exception as e:
        print(f"Error generating user command: {e}")
        # Fallback to a simple description if API call fails
        return f"Process G-code for {os.path.basename(file_path)}"

def write_to_jsonl(file_handle, prompt, completion):
    """Write a JSON line with prompt and completion fields to the file."""
    # Create the JSON object
    json_obj = {
        "prompt": prompt,
        "completion": completion
    }
    
    # Write the JSON line to the file
    file_handle.write(json.dumps(json_obj) + "\n")
    
    # Return 1 for the line count
    return 1

def main(train_lines=100000, dev_lines=10000, train_file="train.jsonl", dev_file="dev.jsonl", dev_ratio=0.1):
    """Main function to download G-code and save to training and development files."""
    ds = load_dataset("bigcode/the-stack-v2", "G-code", streaming=True, split="train")
    
    train_total_lines = 0
    dev_total_lines = 0
    processed_files = 0
    
    # Create output directories if they don't exist
    for file_path in [train_file, dev_file]:
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
    
    with open(train_file, 'w', encoding='utf-8') as train_f, open(dev_file, 'w', encoding='utf-8') as dev_f:
        for sample in ds:
            repo_name = sample['repo_name']
            file_path = sample['path']
            
            print(f"Processing file {processed_files+1}:")
            print(f"  Path: {file_path}")
            print(f"  Repository: {repo_name}")
            
            gcode_content = download_gcode_content(repo_name, file_path)
            
            if gcode_content:
                # Generate user command
                user_command = generate_user_command(repo_name, file_path, gcode_content)
                
                if dev_total_lines < dev_lines and (train_total_lines >= train_lines or 
                                                   random.random() < dev_ratio):
                    write_to_jsonl(dev_f, user_command, gcode_content)
                    dev_total_lines += 1
                    print(f"  Added to dev set (Total dev: {dev_total_lines})")
                else:
                    write_to_jsonl(train_f, user_command, gcode_content)
                    train_total_lines += 1
                    print(f"  Added to train set (Total train: {train_total_lines})")
            else:
                print(f"  Failed to download content")
            
            processed_files += 1
            print()
            
            # Check if we've reached the desired number of lines for both sets
            if train_total_lines >= train_lines and dev_total_lines >= dev_lines:
                print(f"Reached targets: Train {train_lines} examples, Dev {dev_lines} examples. Stopping.")
                break
    
    print(f"Completed. Processed {processed_files} files.")
    print(f"Training set: {train_total_lines} examples in {train_file}")
    print(f"Development set: {dev_total_lines} examples in {dev_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download G-code files for training and development data')
    parser.add_argument('--train-lines', type=int, default=100000, 
                        help='Target number of examples for training set (default: 100000)')
    parser.add_argument('--dev-lines', type=int, default=10000, 
                        help='Target number of examples for development set (default: 10000)')
    parser.add_argument('--train-file', type=str, default='data/train.jsonl',
                        help='Training output file path (default: data/train.jsonl)')
    parser.add_argument('--dev-file', type=str, default='data/dev.jsonl',
                        help='Development output file path (default: data/dev.jsonl)')
    parser.add_argument('--dev-ratio', type=float, default=0.1,
                        help='Ratio of files to allocate to dev set (default: 0.1)')
    
    args = parser.parse_args()
    main(train_lines=args.train_lines, 
         dev_lines=args.dev_lines, 
         train_file=args.train_file, 
         dev_file=args.dev_file,
         dev_ratio=args.dev_ratio)
    

    # Example usage: 
    # python data\stacks_gcode_jsonl.py --train-lines 10000 --dev-lines 1000 --train-file data\train_10000.jsonl --dev-file data\dev_1000.jsonl --dev-ratio 0.1