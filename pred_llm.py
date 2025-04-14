from datasets import load_dataset
import os
import json
from openai import OpenAI
from markitdown import MarkItDown
from transformers import AutoTokenizer
import argparse


def main(args):
    PROMPT='I am writing a scientific paper with the title:\n{title}\n Here is the abstract of that paper:\n{abstract}\n I have gathered related work section of similar previous papers in the literature as follows. Related Papers:\n\n{citation}\n\n Please help me write write one paragraph of related work, also known as literature review, for that paper that contrasts my work from the previous papers listed in the context above. Related Work:\n'  #regarding Improving Reasoning Through Aggregation.
    PROMPT_V2 ='I am writing a scientific paper with the title:\n{title}\n Here is the abstract of that paper:\n{abstract}\n I have gathered title and related work section of similar previous papers in the literature as follows. Related Papers:\n\n{citation}\n\n Please help me write write one paragraph of related work, also known as literature review, for that paper that contrasts my work from the previous papers listed in the context above. Related Work:\n'
    PROMPT_V3 ='I am writing a scientific paper with the title:\n{title}\n Here is the abstract of that paper:\n{abstract}\n I have gathered title and related work section and contributions of similar previous papers in the literature as follows. Related Papers:\n\n{citation}\n\n Please help me write write one paragraph of related work, also known as literature review, for that paper that contrasts my work from the previous papers listed in the context above. Related Work:\n'
    PROMPT_V4 ='I am writing a scientific paper with the title:\n{title}\n Here is the abstract of that paper:\n{abstract}\n I have gathered title and related work section and contributions of similar previous papers in the literature as follows. Related Papers:\n\n{citation}\n\n Please help me write write one paragraph of related work for that paper that contrasts my work from the previous papers listed in the context above regarding Improving Reasoning by Aggregation. Related Work:\n'
    PROMPT_V5 = 'I am writing a scientific paper with the title:\n{title}\n Here is the abstract of that paper:\n{abstract}\n I have gathered title and related work section and contributions of similar previous papers in the literature as follows. Related Papers:\n\n{citation}\n\n Please help me write one paragraph of related work for that paper that contrasts my work from the previous papers listed in the context above regarding Improving Reasoning by Aggregation. Please plan out the paragraph writing task as follows:\n 1.First identify the sub-topics on which you want to contrast the current paper vs the previous one.\n2.Differences to highlight about the current work vs the previous work for each of them.\n3.Write out the above points in a paragraph form and be sure to always cite the previous work when drawing the contrast with the current work. Related Work:\n'
    PROMPT_V6 = 'I am writing a scientific paper with the title:\n{title}\n Here is the abstract of that paper:\n{abstract}\n I have gathered title and related work section and contributions of similar previous papers in the literature as follows. Related Papers:\n\n{citation}\n\n Please help me write one paragraph of related work for that paper that contrasts my work from the previous papers listed in the context above regarding Improving Reasoning by Aggregation. Please plan out the paragraph writing task as follows:\n 1.First identify highlight of each papers mentioned in the citation list on which you want to contrast the current paper with this template Title:highlight. If you do not want to cite the paper, provide the title but leave the highlight section empty.\n2.Write out the above points in a paragraph form and be sure to always cite using a number that coressponds to the order in the list of papers in the first step. paragraph should be self contained and mainly talk about previous works. At the end of the paragraph add a few senteces explaining the contrast from this work.\n'
    GET_RW_PROMPT = 'Please extract the related work section of the paper below.\nPaper:\n\n{paper}\n\n Do not paraphrase or summarize. Do not include any other text. Just output the original related work section of the paper. This section might have slightly different name such as literature review. Related Work:\n'
    GET_TITLE_PROMPT = 'Please extract the title of the paper below.\nPaper:\n\n{paper}\n\n Do not paraphrase or summarize. Do not include any other text. Just output the original title of the paper. Title:\n'
    GET_CONTRIB_PROMPT = 'Please extract main contributions and highlights of the paper below.\nPaper:\n\n{paper}\n\n Use bulletpoints for your response. Do not include any other text. Just output main contributions and highlights of the paper. Contributions and Highlights:\n'
    # PROMPT='I am writing a scientific paper with the title:\n{title}\n Here is the abstract of that paper:\n{abstract}\n Please help me write write one paragraph of related work, also known as literature review, for that paper that contrasts my work from the previous papers. Write the literature review only no extra sentences.'  #regarding Improving Reasoning Through Aggregation.
    API=args.api
    MODEL=args.model
    APPROACH=args.approach
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")
    if API=="CB":
        openai_api_key = "serving-on-vllm"
        openai_api_base = "http://localhost:8000/v1"
        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
            timeout=None,
        )
        model_name=f'{APPROACH}:{MODEL}'
    elif API=="OPENAI":
        openai_api_key = os.getenv('OPENAI_API_KEY')
        openai_api_base = "https://api.openai.com/v1"
        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
            timeout=None,
        )
        model_name=MODEL
    elif API=="TOGETHER":
        openai_api_key = os.getenv('TOGETHER_API_KEY')
        openai_api_base = "https://api.together.xyz/v1"
        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
            timeout=None,
        )
        model_name=MODEL
    # output_list = []
    md = MarkItDown()
    paper_id = '2409.12147v1'
    paper_md = md.convert(f'papers/{paper_id}.pdf')
    paper_txt = paper_md.text_content
    abstract = paper_txt.split('\n\nABSTRACT\n\n')[1].split('\n\n1\n\nINTRODUCTION\n\n')[0]
    title = 'MAgICoRe: Multi-Agent, Iterative, Coarse-to-Fine Refinement for Reasoning'
    citation_path = [f for f in os.listdir('citations/') if os.path.isfile(os.path.join('citations/', f))]
    citation_rw_list = []
    citation_title_list = []
    citation_contrib_list = []
    # breakpoint()
    for i,p in enumerate(citation_path):
        if True: #i < 8:
            ref = md.convert(f'citations/{p}')
            ref_txt = ref.text_content
            get_rw_prompt = GET_RW_PROMPT.format(paper=ref_txt)
            get_title_prompt = GET_TITLE_PROMPT.format(paper=ref_txt)
            get_contrib_prompt = GET_CONTRIB_PROMPT.format(paper=ref_txt)
            messages = [
                    {
                        "role": "user",
                        "content": get_rw_prompt
                    },
                ]
            completion = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                    )
            citation_rw_list.append(completion.choices[0].message.content)
            messages = [
                    {
                        "role": "user",
                        "content": get_title_prompt
                    },
                ]
            completion = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                    )
            citation_title_list.append(completion.choices[0].message.content)
            messages = [
                    {
                        "role": "user",
                        "content": get_contrib_prompt
                    },
                ]
            completion = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                    )
            citation_contrib_list.append(completion.choices[0].message.content)
    related_work = paper_txt.split('RELATED WORK\n\n')[1].split('CONCLUSION\n\n')[0]
    citation_list = [' : '.join((a,b,c)) for a,b,c in zip(citation_title_list,citation_rw_list,citation_contrib_list)]
    prompt = PROMPT_V6.format(title=title,abstract=abstract,citation='\n\n'.join(citation_list))
    messages = [
                    {
                        "role": "user",
                        "content": prompt
                    },
                ]
    tok_prompt = tokenizer(prompt)
    seq_len = len(tok_prompt['input_ids'])
    print(seq_len)
    completion = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                    )
    prediction = completion.choices[0].message.content
    print(prediction)
    with open(args.output_path, 'w') as fout:
        json.dump({'citations_rw':citation_rw_list,'citations_title':citation_title_list,'citations_contrib':citation_contrib_list,'generated_related_work':prediction} , fout)


    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Generate model responses on ScholarQA')
    parser.add_argument('--api', required=True, help='Name of the api provider')
    parser.add_argument('--model', required=True, help='Name of the model')
    parser.add_argument('--approach',  help='Approach for CB api')
    parser.add_argument('--iteration',  help='Iteration')
    parser.add_argument('--output_path', required=True, help='directory to save the output JSON file')
    args = parser.parse_args()


    main(args)