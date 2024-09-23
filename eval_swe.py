import os

import openai

from moatless.index import IndexSettings
from moatless.edit import EditCode, PlanToCode
from moatless.find import DecideRelevance, IdentifyCode, SearchCode
from moatless.transitions import search_and_code_transitions
from moatless.benchmark.evaluation import Evaluation


api_key = "alignment-data-generation"
base_url = "https://api.01ww.xyz/v1"
os.environ['OPENAI_API_KEY'] = api_key
os.environ['OPENAI_API_BASE'] = base_url
openai.api_key = api_key
openai.base_url = base_url


def main():
    model = "gpt-4o-2024-05-13"
    temperature = 0.0

    max_cost=0.5

    index_settings = IndexSettings(
        embed_model="text-embedding-3-small"
    )

    global_params = {
        "model": model,
        "temperature": temperature,
        "max_tokens": 2000,
        "max_prompt_file_tokens": 12000,
    }

    state_params = {
        SearchCode: {
            "max_search_results": 75,
            "provide_initial_context": True,  # Do a vector search with the problem statement to get an initial file context
            "initial_context_tokens": 6000,
            "initial_search_results": 100,
            "initial_context_spans_per_file": 5,
        },
        IdentifyCode: {
            "expand_context": True,  # Expands the search results with related code to the search hits
        },
        DecideRelevance: {
            "finish_after_relevant_count": 1,  # Even if the LLM doesn't believe the identified code is complete we will finish up after one retry
        },
        PlanToCode: {
            "max_tokens_in_edit_prompt": 750, # The max number of tokens in the edit block
            "expand_context_with_related_spans": False,
            "finish_on_review": True, # To abort if the LLm suggest reviews of the code, it's only possible to apply changes ATM.
        },
        EditCode: {
            "chain_of_thought": False,
            "show_file_context": False,
            "max_prompt_file_tokens": 8000,
        },
    }

    index_store_dir = "./index_store"
    evaluations_dir = "./evaluations"
    evaluation_name = "20240922_moatless_claude35sonnet"

    instance_whitelist = [
        "pytest-dev__pytest-5227", 
        # "sympy__sympy-24152", 
        # "django__django-16139",
        # "django__django-16379", 
        # "django__django-16527"
    ]

    search_and_code = search_and_code_transitions(
        global_params=global_params, state_params=state_params
    )
    evaluation = Evaluation(
        index_settings=index_settings,
        index_store_dir=index_store_dir,
        repo_base_dir="/home/yifan/Code/moatless-tools/repos",  # must be absolute path, maybe some bug in LlamaIndex
        evaluations_dir=evaluations_dir,
        evaluation_name=evaluation_name,
        transitions=search_and_code,
        max_cost=max_cost,
        max_file_context_tokens=16000
    )

    evaluation.run_swebench_evaluation(instance_ids=instance_whitelist)

if __name__ == "__main__":
    main()
