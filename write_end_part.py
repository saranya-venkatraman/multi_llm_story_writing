import pandas as pd
import re
import numpy as np
import argparse
import logging
import os
from tqdm.notebook import tqdm
from vllm import LLM, SamplingParams
import hf_olmo
import os
import torch
import nltk

nltk.download("punkt")
from transformers import pipeline

os.environ['HF_TOKEN'] = "ENTER HUGGINGFACE TOKEN HERE"
# Suppressing the loading progress output
pd.options.mode.chained_assignment = None


def main():
    # To parse all the arguments
    parser = argparse.ArgumentParser()

    # Add named arguments
    parser.add_argument(
        "--author_num",
        type=int,
        help="The order of this llm or part of story being written",
    )
    parser.add_argument("--start", type=int, help="prompt start index", default=0)
    parser.add_argument("--end", type=int, help="prompt end index", default=-1)
    parser.add_argument("--n", type=int, help="batch_size")
    parser.add_argument("--total_authors", type=int)
    parser.add_argument("--llm", type=str, help="LLM being used to generate story")

    # Parse the command-line arguments
    args = parser.parse_args()
    author_num = args.author_num
    start = args.start
    end = args.end
    n = args.n
    total_authors = args.total_authors
    llm = args.llm

    # Output file
    out_file = (
        "./generated_stories{}/author_{}_{}.csv".format(
            total_authors, author_num, llm
        )
    )

    # To get number of words in a story
    def get_len(story):
        return len(story.split())

    # Function to get story from HuggingFace LLMs
    def get_story_from_hf_llm(data_df, ixs, num_attempts):
        if num_attempts == 0:
            ixs = data_df.index
            df = data_df
        else:
            df = data_df.loc[ixs]

        prompts, summaries, last_sents = [], [], []
        for index, row in df.iterrows():
            story_so_far = row["full_story"]
            summary = summarizer(
                story_so_far, max_length=100, min_length=30, do_sample=False
            )
            sentences_ = nltk.sent_tokenize(story_so_far)
            last_story_sentence = sentences_[-1]
            summarized_story = summary[0]["summary_text"]
            prompt = "Write 180 words \
                 to conclude this storyline: {}. Do not add any instructions. Continue from this sentence: {}".format(
                summarized_story, last_story_sentence
            )
            # prompt = 'You are a creative story writer. You write very long stories. Write at least 300 words to continue this storyline: {}. Do not add anything, start wirting immediately. Continue from this sentence: {}'.format(summarized_story, last_story_sentence)
            prompts.append(prompt)
            summaries.append(summarized_story)
            last_sents.append(last_story_sentence)
            # Generate texts from the prompts. The output is a list of RequestOutput objects
            # that contain the prompt, generated text, and other information.

        outputs = llm_chosen.generate(prompts, sampling_params)
        # Print the outputs.
        stories = []
        for i in range(len(outputs)):
            prompt = outputs[i].prompt
            generated_text = outputs[i].outputs[0].text
            story = re.sub(" +", " ", generated_text)
            story = re.sub(re.escape(last_sents[i]), "", generated_text)
            story = re.sub(
                "Write 180 words.*?from this sentence:", "", story, flags=re.DOTALL
            )
            story = re.sub(re.escape(prompt), "", story, flags=re.DOTALL)
            sentences = nltk.sent_tokenize(story)
            # Remove incomplete sentence from the end
            try:
                last_sentence = sentences[-1]
                words = nltk.word_tokenize(last_sentence)
                if words[0][-1] not in ".!?" or "end" in last_sentence:
                    sentences.pop()
                # Join the remaining sentences to form the final text
                cleaned_story = " ".join(sentences)
            except:
                cleaned_story = ""
            stories.append(cleaned_story)

        df[story_column] = stories
        df["story_summary_{}".format(author_num - 1)] = summaries
        df["part{}_words".format(author_num)] = df.apply(
            lambda x: get_len(x[story_column]), axis=1
        )

        short_ix = df.index[df["part{}_words".format(author_num)] < 160]
        if len(short_ix) > 9 and num_attempts < 20:
            num_attempts += 1
            data_df.loc[ixs] = get_story_from_hf_llm(
                df, ixs=short_ix, num_attempts=num_attempts
            )
            # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        return data_df

    def generate_batches_of_n_indices(start, end, num_samples):
        for i in range(start, end + 1, num_samples):
            yield i, min(i + num_samples, end)

    # This is the input file format that contains all the prompts, author order information and the
    # story parts generated so far. After each story part is written, the "full_story" column is updated with all the parts so far.
    # This column is sent to the summarizer, hence it is updated after each new part is addded to the story.
    # There is a different script to help generate this combined file  in this format and example files with this format
    # are in the "/combined_files" folder

    combined_file = (
        "./generated_stories{}/{}_part{}.csv".format(
            total_authors, total_authors, author_num - 1
        )
    )
    df = pd.read_csv(combined_file)
    # df.set_index(['index','old_index', 'author_order']).index.is_unique

    # Get slice of DataFrame for which first author is "llm"
    llm_df = df[df["author{}".format(author_num)] == llm].reset_index()
    # Use all data, if not using a subset of input data, i.e. when default = -1
    if end == -1:
        end = len(llm_df)
    llm_df = llm_df[:end]

    llms_to_model_dict = {
        "llama": "meta-llama/Llama-2-13b-chat-hf",
        "olmo": "allenai/OLMo-7B-Instruct",
        "gemma": "google/gemma-1.1-7b-it",
        "mixtral": "mistralai/Mistral-7B-Instruct-v0.2",
        "orca": "microsoft/Orca-2-13b",
    }
    print("LLM to Model mapping: ", llms_to_model_dict)

    # List of HuggingFace and non-HF models
    list_of_hf_llms = list(df["author{}".format(author_num)].unique())

    # Picking only indexing columns and actual story part generated to reduce data storage redundancy
    # columns_to_write = ['index','old_index', 'author_order', 'part{}'.format(author_num)]

    # For HuggingFace LLMs
    if llm in list_of_hf_llms:
        summarizer = pipeline("summarization", model="Falconsai/text_summarization")
        # Get corresponding model
        model_chosen = llms_to_model_dict[llm]
        print("Chosen model", model_chosen)

        # max_tokens need to be adjusted based on value of N (number of authors, or number of words per author)
        sampling_params = SamplingParams(top_p=0.9, top_k=50, max_tokens=260)
       

        # Create an LLM.
        llm_chosen = LLM(model=model_chosen, dtype="half")

        # Process data in batches and write one batch at a time
        for batch_start, batch_end in generate_batches_of_n_indices(start, end, n):
            start_ix, end_ix = batch_start, batch_end
            temp_df = llm_df[start_ix:end_ix]
            story_column = "part{}".format(author_num)
            temp_df[story_column] = ""
            temp_df = get_story_from_hf_llm(temp_df, ixs=None, num_attempts=0)
            with open(out_file, "a") as f:
                temp_df.to_csv(f, header=f.tell() == 0, index=False, encoding="utf-8")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
