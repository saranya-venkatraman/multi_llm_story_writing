

import re
import pandas as pd
from glob import glob

# First cleaning all the parts so far and combining it for next part of story to be generated

def make_final_file(llm_name, num_authors=5):
    file = "./generated_stories{}/author_4_{}.csv".format(num_authors, llm_name)

    df = pd.read_csv(file)
    print(df.columns)
   
    columns_to_keep = ['prompt', 'story', 'num_words', 'author1',
       'author2', 'author3', 'author4', 'author5', 'author_order', 'part1',
       'part1_words', 'part2', 'part2_words', 'story_summary_1', 'part3',
       'story_summary_2', 'part3_words', 'part4',
       'story_summary_3', 'part4_words']
            
    df = df[columns_to_keep]
    df.dropna(subset=["part1"], inplace=True)
    df.dropna(subset=["part2"], inplace=True)
    df.dropna(subset=["part3"], inplace=True)
    df.dropna(subset=["part4"], inplace=True)

    def remove_extra_spaces_and_prompt(prompt, story):
        pattern = r'\*{1,3}.*?\*{1,3}'
        story = story.replace(prompt,'')
        story = re.sub(pattern, '', story)
        story = re.sub(r'\.{2,}', '', story)
        story = re.sub(r'<newline> <newline>', '', story)
        story = re.sub('Write 180 words.*?from this sentence:','',story, flags=re.DOTALL)
        story = re.sub(' +', ' ', story)
        sentences = story.split(". ")
        story = ". ".join(s for s in sentences if not s.strip().startswith("-"))
        story ="".join(s for s in story.splitlines() if not s.startswith("##"))
        return story
    # Applying it to two columns
    df['part1'] = df.apply(lambda x: remove_extra_spaces_and_prompt(x['prompt'], x['part1']), axis=1)
    df['part2'] = df.apply(lambda x: remove_extra_spaces_and_prompt(x['prompt'], x['part2']), axis=1)
    df['part3'] = df.apply(lambda x: remove_extra_spaces_and_prompt(x['prompt'], x['part3']), axis=1)
    df['part4'] = df.apply(lambda x: remove_extra_spaces_and_prompt(x['prompt'], x['part4']), axis=1)


    df = df[df["part1"].str.contains("no story")==False]
    df = df[df["part2"].str.contains("no story")==False]
    df = df[df["part3"].str.contains("no story")==False]
    df = df[df["part4"].str.contains("no story")==False]


    def get_len(story):
        return len(story.split())
    # Applying it to two columns
    df["part1_words"] = df.apply(lambda x: get_len(str(x['part1'])), axis=1)
    df["part2_words"] = df.apply(lambda x: get_len(str(x['part2'])), axis=1)
    df["part3_words"] = df.apply(lambda x: get_len(str(x['part3'])), axis=1)
    df["part4_words"] = df.apply(lambda x: get_len(str(x['part4'])), axis=1)

    df["full_story"] = df.apply(lambda x: x['part1']+x['part2']+x['part3']+x['part4'], axis=1)
    df = df.reset_index()

    df_story = df.loc[(df['part1_words'] >= 50) & (df['part2_words'] >=50) & \
     (df['part3_words'] >=50) & (df['part4_words'] >=50)]

    print(len(df_story))
    # # print(df_story.columns)
    # print(df_story.head())

    df_story.to_csv("./generated_stories{}/4_{}.csv".format(num_authors, llm_name))

    return


make_final_file("gemma")
make_final_file("llama")
make_final_file("olmo")
make_final_file("mixtral")
make_final_file("orca")


list_of_files = glob("./generated_stories5/4*.csv")
print(list_of_files)
dfs = [pd.read_csv(f) for f in list_of_files]
df = pd.concat(dfs)
print(df.columns)

columns_to_keep = ['prompt', 'story', 'num_words', 'author1',
       'author2', 'author3', 'author4', 'author5', 'author_order', 'part1',
       'part1_words', 'part2', 'part2_words', 'story_summary_1', 'part3',
       'story_summary_2', 'part3_words', 'part4',
       'story_summary_3', 'part4_words', 'full_story']


df = df[columns_to_keep]

df.to_csv("./combined_files/5_part4.csv")

