import os
import logging
import subprocess

import openai

from moatless import AgenticLoop
from moatless.index import CodeIndex, IndexSettings
from moatless import FileRepository, Workspace
from moatless.transitions import search_transitions, code_transitions, search_and_code_transitions


api_key = "alignment-data-generation"
base_url = "https://api.01ww.xyz/v1"
os.environ['OPENAI_API_KEY'] = api_key
os.environ['OPENAI_API_BASE'] = base_url
openai.api_key = api_key
openai.base_url = base_url


def reset_repo(repo_dir):
    try:
        subprocess.run(
            ["git", "checkout", "."],
            cwd=repo_dir,
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        raise e


def run_search_loop(workspace, model, instructions):
    search_loop = AgenticLoop(
        transition_rules=search_transitions(model=model), 
        workspace=workspace,
        initial_message=instructions,
        prompt_log_dir="../logs",
    )

    search_response = search_loop.run()

    print(workspace.file_context.create_prompt())

    return search_response


def run_code_loop(workspace, model, instructions, repo_dir):
    code_loop = AgenticLoop(
        transition_rules=code_transitions(model=model),
        workspace=workspace,
        prompt_log_dir="../logs",
    )
    code_response = code_loop.run(instructions)

    print(f"Response: {code_response.message}")


def run_search_and_code_loop(workspace, model, instructions, repo_dir):
    search_and_code_loop = AgenticLoop(
        transition_rules=search_and_code_transitions(model=model),
        workspace=workspace,
        prompt_log_dir="../logs",
    )
    search_and_code_response = search_and_code_loop.run(instructions)

    print(f"Response: {search_and_code_response.message}")


def output_diff(repo_dir):
    output = subprocess.run(
        ["git", "diff"],
        capture_output=True,
        text=True,
        cwd=repo_dir,
    )

    print(output.stdout)


def main():
    model = "gpt-4o-2024-05-13"
    index_settings = IndexSettings(
        embed_model="text-embedding-3-small"
    )

    repo_dir = "/home/yifan/Code/test_repo"
    reset_repo(repo_dir)
    file_repo = FileRepository(repo_path=repo_dir)

    code_index = CodeIndex(file_repo=file_repo, settings=index_settings)
    nodes, tokens = code_index.run_ingestion()

    print(f"Indexed {nodes} nodes and {tokens} tokens")

    workspace = Workspace(file_repo=file_repo, code_index=code_index)

    instructions = "Remove the token limit check from the completion function"

    search_response = run_search_loop(workspace, model, instructions)
    
    run_code_loop(workspace, model, search_response.message, repo_dir)

    output_diff(repo_dir)


if __name__ == "__main__":
    logger = logging.getLogger("Loop")
    if not logger.handlers:
        print("Logger has no handlers, adding one.")
        console_handler = logging.StreamHandler()
        logger.addHandler(console_handler)  # add stdout handler for Loop logger
    logging.getLogger("Loop").setLevel(logging.INFO)

    main()
