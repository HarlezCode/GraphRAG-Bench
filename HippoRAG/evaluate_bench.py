import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from typing import List
import json

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig
import pandas as pd
import logging

def get_gold_docs(samples: List, dataset_name: str = None) -> List:
    gold_docs = []
    for sample in samples:
        if 'supporting_facts' in sample:  # hotpotqa, 2wikimultihopqa
            gold_title = set([item[0] for item in sample['supporting_facts']])
            gold_title_and_content_list = [item for item in sample['context'] if item[0] in gold_title]
            if dataset_name.startswith('hotpotqa'):
                gold_doc = [item[0] + '\n' + ''.join(item[1]) for item in gold_title_and_content_list]
            else:
                gold_doc = [item[0] + '\n' + ' '.join(item[1]) for item in gold_title_and_content_list]
        elif 'contexts' in sample:
            gold_doc = [item['title'] + '\n' + item['text'] for item in sample['contexts'] if item['is_supporting']]
        else:
            assert 'paragraphs' in sample, "`paragraphs` should be in sample, or consider the setting not to evaluate retrieval"
            gold_paragraphs = []
            for item in sample['paragraphs']:
                if 'is_supporting' in item and item['is_supporting'] is False:
                    continue
                gold_paragraphs.append(item)
            gold_doc = [item['title'] + '\n' + (item['text'] if 'text' in item else item['paragraph_text']) for item in gold_paragraphs]

        gold_doc = list(set(gold_doc))
        gold_docs.append(gold_doc)
    return gold_docs


def get_gold_answers(samples):
    gold_answers = []
    for sample_idx in range(len(samples)):
        gold_ans = None
        sample = samples[sample_idx]

        if 'answer' in sample or 'gold_ans' in sample:
            gold_ans = sample['answer'] if 'answer' in sample else sample['gold_ans']
        elif 'reference' in sample:
            gold_ans = sample['reference']
        elif 'obj' in sample:
            gold_ans = set(
                [sample['obj']] + [sample['possible_answers']] + [sample['o_wiki_title']] + [sample['o_aliases']])
            gold_ans = list(gold_ans)
        assert gold_ans is not None
        if isinstance(gold_ans, str):
            gold_ans = [gold_ans]
        assert isinstance(gold_ans, list)
        gold_ans = set(gold_ans)
        if 'answer_aliases' in sample:
            gold_ans.update(sample['answer_aliases'])

        gold_answers.append(gold_ans)

    return gold_answers

def readCorpus(path):
    docs = []

    old_path = os.path.join(path, "textbook")
    for i in range(20):
        path = old_path + str(i+1) + f"/textbook{i+1}_structured.json"
        corpus = pd.read_json(path)
        for _, row in corpus.iterrows():
            docs.append(row['chapter'] + ": " + row['section'] + "," + row['subsection'] + ","  + row['subsubsection'] + '\n' +row['content']) 
    return docs

def readQueries(path):
    df = pd.read_json(path, lines=True, orient="records")
    if ('TF' in path):
        return [row['Question'] + " Only give a boolean answer, False or True." for row in df.iloc]
    return [row['Question'] for row in df.iloc]

    




def main():
    save_dir = 'outputs'# Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
    # llm_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    llm_model_name = "microsoft/Phi-3-mini-128k-instruct" 

    embedding_model_name = "nvidia/NV-Embed-v2" # Embedding model name (NV-Embed, GritLM or Contriever for now)
    llm_base_url= "http://localhost:8000/v1" # Base url for your deployed LLM (i.e. http://localhost:8000/v1)



    docs = readCorpus('./dataset/corpus/')
    all_queries = readQueries('./dataset/questions/TF.jsonl')




    config = BaseConfig(
        save_dir=save_dir,
        llm_base_url=llm_base_url,
        llm_name=llm_model_name,
        embedding_model_name=embedding_model_name,
        force_index_from_scratch=False,  # ignore previously stored index, set it to False if you want to use the previously stored index and embeddings
        force_openie_from_scratch=False,
        # rerank_dspy_file_path="src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
        retrieval_top_k=20,
        linking_top_k=5,
        max_qa_steps=3,
        qa_top_k=5,
        embedding_batch_size=4,
        max_new_tokens=None,
        corpus_len=len(docs),
        openie_mode='online',
        max_retry_attempts=20
    )

    logging.basicConfig(level=logging.INFO)

    hipporag = HippoRAG(global_config=config)

    hipporag.index(docs)


    print(len(all_queries))
    # Retrieval and QA
    retrieval_results = hipporag.retrieve(queries=all_queries, num_to_retrieve=1)
    qa_results = hipporag.rag_qa(retrieval_results)
    with open('./results/qa_20_filter_TF.txt', 'a', encoding='utf-8') as file:
        file.write(str(qa_results))

    print(len(qa_results))
    qa_results2 = [i.to_dict() for i in qa_results[0]]

    # Write the list of dictionaries to the JSON file
    with open("./results/qa_20_filter_TF.json", "a", encoding='utf-8') as f:
        json.dump(qa_results2, f, indent=4) 
        



if __name__ == "__main__":
    main()
