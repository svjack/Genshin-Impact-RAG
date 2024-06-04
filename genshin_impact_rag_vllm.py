'''
pip install llama-cpp-python
pip install sentence_transformers
pip install pandas
pip install langchain
pip install faiss-cpu
pip install huggingface_hub
pip install vllm
'''

'''
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen1.5-7B-Chat-AWQ --dtype auto \
 --api-key token-abc123 --quantization awq --max-model-len 6000 --gpu-memory-utilization 0.9
'''

from openai import OpenAI

import gradio as gr

name_client_dict = {
    "chat": {
    "model": "Qwen/Qwen1.5-7B-Chat-AWQ",
    "client": OpenAI(
        api_key="token-abc123",
        base_url="http://localhost:8000/v1",
    )}
}

def openai_predict(messages,
    adapter_name = "chat",
    temperature = 0.01,
    response_format = None
    ):
    if response_format is None:
        stream = name_client_dict[adapter_name]["client"].chat.completions.create(
                model=name_client_dict[adapter_name]["model"],  # Model name to use
                messages=messages,  # Chat history
                temperature=temperature,  # Temperature for text generation
                stream=True,  # Stream response
        )
    else:
        stream = name_client_dict[adapter_name]["client"].chat.completions.create(
                model=name_client_dict[adapter_name]["model"],  # Model name to use
                messages=messages,  # Chat history
                temperature=temperature,  # Temperature for text generation
                stream=True,  # Stream response
                response_format = response_format
        )

    # Read and return generated text from response stream
    partial_message = ""
    for chunk in stream:
        #clear_output(wait = True)
        partial_message += (chunk.choices[0].delta.content or "")
        #print(partial_message)
        yield partial_message
    #return partial_message

import pandas as pd
import numpy as np
import os
import json

from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain import chains

from huggingface_hub import snapshot_download

if not os.path.exists("genshin_book_chunks_with_qa_sp"):
    path = snapshot_download(
        repo_id="svjack/genshin_book_chunks_with_qa_sp",
        repo_type="dataset",
        local_dir="genshin_book_chunks_with_qa_sp",
        local_dir_use_symlinks = False
    )

if not os.path.exists("bge_small_qq_qa_prebuld"):
    path = snapshot_download(
        repo_id="svjack/bge_small_qq_qa_prebuld",
        repo_type="dataset",
        local_dir="bge_small_qq_qa_prebuld",
        local_dir_use_symlinks = False
    )

'''
import llama_cpp
import llama_cpp.llama_tokenizer

llama = llama_cpp.Llama.from_pretrained(
    repo_id="Qwen/Qwen1.5-14B-Chat-GGUF",
    filename="*q4_0.gguf",
    tokenizer=llama_cpp.llama_tokenizer.LlamaHFTokenizer.from_pretrained("Qwen/Qwen1.5-14B"),
    verbose=False,
    n_gpu_layers = -1,
    n_ctx = 3060
)
'''

qst_qq_qa_mapping_df = pd.read_csv("genshin_book_chunks_with_qa_sp/genshin_qq_qa_mapping.csv").dropna()
qst_qq_qa_mapping_df

texts = qst_qq_qa_mapping_df["emb_text"].dropna().drop_duplicates().values.tolist()
len(texts)

embedding_path = "svjack/bge-small-qq-qa"
bge_qq_qa_embeddings = HuggingFaceEmbeddings(model_name=embedding_path,
                                            model_kwargs = {'device': 'cpu'}
                                            )

docsearch_qq_qa_loaded = FAISS.load_local("bge_small_qq_qa_prebuld/", bge_qq_qa_embeddings,
                                         allow_dangerous_deserialization = True
                                         )

def uniform_recall_docs_to_pairwise_cos(query ,doc_list, embeddings):
    assert type(doc_list) == type([])
    from langchain.evaluation import load_evaluator
    from langchain.evaluation import EmbeddingDistance
    hf_evaluator = load_evaluator("pairwise_embedding_distance", embeddings=embeddings,
                              distance_metric = EmbeddingDistance.COSINE)
    return sorted(pd.Series(doc_list).map(lambda x: x[0].page_content).map(lambda x:
        (x ,hf_evaluator.evaluate_string_pairs(prediction=query, prediction_b=x)["score"])
    ).values.tolist(), key = lambda t2: t2[1])

def recall_df_to_prompt_info_part(recall_df):
    cdf = recall_df[
        recall_df["source"] == "character"
    ]
    bdf = recall_df[
        recall_df["source"] == "book"
    ]
    req = []
    if cdf.size > 0:
        l = cdf.apply(
            lambda x:
                "问题：{}\n 答案：{}".format(x["emb_text"], x["out_text"])
            , axis = 1
        ).values.tolist()
        req.append(
        '''
        下面是有关游戏角色的问答信息：
        {}
        '''.format(
            "\n\n".join(l)
        ).strip().replace("\n\n\n", "\n\n")
        )
    if bdf.size > 0:
        l = bdf.apply(
            lambda x:
                "{}".format(x["out_text"])
            , axis = 1
        ).values.tolist()
        req.append(
        '''
        下面是有关游戏设定的介绍信息：
        {}
        '''.format(
            "\n\n".join(l)
        ).strip().replace("\n\n\n", "\n\n")
        )
    req = "\n\n".join(req).strip().replace("\n\n\n", "\n\n")
    req = "\n".join(map(lambda x: x.strip() ,req.split("\n")))
    return req

def produce_problem_context_prompt(query, k = 10):
    t2_list = uniform_recall_docs_to_pairwise_cos(
        query,
        docsearch_qq_qa_loaded.similarity_search_with_score(query, k = k, ),
        bge_qq_qa_embeddings,
    )
    if t2_list:
        out = pd.DataFrame(t2_list).apply(
            lambda x:
            qst_qq_qa_mapping_df[
            qst_qq_qa_mapping_df["emb_text"] == x.iloc[0]
            ].apply(lambda y:
                dict(list(y.to_dict().items()) + [("score", x.iloc[1])]), axis = 1
            ).values.tolist()
            , axis = 1
        )
        out = pd.DataFrame(out.explode().dropna().values.tolist())
        out = recall_df_to_prompt_info_part(out)
        return "根据下面提供的游戏角色信息及游戏设定信息回答问题：{}".format(query) + "\n" + \
        "注意从提供的信息中进行甄选，只使用与问题相关的信息回答问题，并给出如此回答的理由及根据。" + "\n\n" + out
    else:
        return "根据下面提供的游戏角色信息及游戏设定信息回答问题：{}".format(query)

from pydantic import BaseModel, Field
class QA(BaseModel):
    需要回答的问题: str = Field(..., description="需要回答的问题")
    给出的答案: str = Field(..., description="给出的答案")
    给出此答案的理由及根据: str = Field(..., description="给出此答案的理由及根据")

def run_problem_context_prompt_once(query):
    #from IPython.display import clear_output
    prompt = produce_problem_context_prompt(query)
    response = openai_predict([{
                "role": "user",
                "content": prompt[:3000]
            }
        ],
        response_format={
        "type": "json_object",
        "schema": QA.schema(),
    },)
    for text in response:
        pass 
    return json.loads(text)
    '''
    response = llama.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": prompt[:3000]
            }
        ],
        response_format={
        "type": "json_object",
        "schema": QA.schema(),
    },
        stream=False,
    )
    '''
    return json.loads(response["choices"][0]["message"]["content"])

run_problem_context_prompt = run_problem_context_prompt_once

def produce_problem_context_prompt_in_character_manner(
    character_name,
    character_setting,
    character_story,
    query,
    answer
):
    character_prompt = '''
    你是一款角色扮演游戏里面的一个NPC，名字叫做{character_name}。

    下面是一个针对问题“{query}”的回答

    回答：{answer}

    要求你以{character_name}的口吻改写回答，要求符合NPC的角色设定并带有相应身份的口吻。

    有关{character_name}的角色介绍是这样的：
    {character_setting}

    有关{character_name}的角色故事是这样的：
    {character_story}
    '''.format(
        **{
            "character_name": character_name,
            "character_setting": character_setting,
            "character_story": character_story,
            "query": query,
            "answer": answer
        }
    )
    character_prompt = "\n".join(map(lambda x: x.strip(), character_prompt.split("\n")))
    return character_prompt

def run_problem_context_prompt_in_character_manner(
    character_name,
    query
):
    #from IPython.display import clear_output
    json_dict_out = run_problem_context_prompt(query)
    answer = list(filter(lambda t2: "的答案" in t2[0] ,json_dict_out.items()))
    if answer:
        answer = answer[0][1]
        character_setting_df = qst_qq_qa_mapping_df[
            qst_qq_qa_mapping_df.apply(
                lambda x: "character" in x["source"] and \
                character_name in x["emb_text"] and \
                "的角色介绍" in x["emb_text"]
                , axis = 1
            )
        ]
        character_story_df = qst_qq_qa_mapping_df[
            qst_qq_qa_mapping_df.apply(
                lambda x: "character" in x["source"] and \
                character_name in x["emb_text"] and \
                "的角色故事" in x["emb_text"]
                , axis = 1
            )
        ]
        if character_setting_df.size > 0 and character_story_df.size > 0:
            character_setting = character_setting_df["out_text"].iloc[0]
            character_story = character_story_df["out_text"].iloc[0]
            character_prompt = produce_problem_context_prompt_in_character_manner(
                character_name,
                character_setting,
                character_story,
                query,
                answer
            )

            response = openai_predict(
             [
                {
                        "role": "user",
                        "content": character_prompt[:3000]
                    }
             ]
            )
            for text in response:
                yield text
            
            '''
            response = llama.create_chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": character_prompt[:3000]
                    }
                ],
                stream=False,
            )
            return response["choices"][0]["message"]["content"]
            req = ""
            for chunk in response:
                delta = chunk["choices"][0]["delta"]
                if "content" not in delta:
                    continue
                print(delta["content"], end="", flush=True)
                req += delta["content"]
            print()
            #clear_output(wait = True)
            return req
            '''
    return ""

all_characters_in_settings = ['丽莎', '行秋', '优菈', '魈', '五郎', '钟离', '温迪',
'菲谢尔', '诺艾尔', '云堇',
 '班尼特', '安柏', '莫娜', '柯莱', '迪卢克', '绮良良', '阿贝多', '卡维', '提纳里',
 '夏沃蕾', '鹿野院平藏', '辛焱', '八重神子', '凝光', '九条裟罗', '米卡', '刻晴',
 '罗莎莉亚', '甘雨', '砂糖', '坎蒂丝', '枫原万叶', '雷电将军', '白术', '娜维娅',
 '芙宁娜', '莱欧斯利', '迪奥娜', '夜兰', '久岐忍', '菲米尼', '可莉', '流浪者',
 '多莉', '凯亚', '琴', '琳妮特', '荒泷一斗', '神里绫人', '夏洛蒂', '雷泽', '芭芭拉',
 '珊瑚宫心海', '妮露', '七七', '香菱', '珐露珊', '赛诺', '神里绫华', '申鹤', '瑶瑶',
 '达达利亚', '早柚', '北斗', '重云', '林尼', '埃洛伊', '托马', '纳西妲', '烟绯',
 '那维莱特', '迪希雅', '宵宫', '胡桃', '艾尔海森', '莱依拉']

if __name__ == "__main__":
    out = run_problem_context_prompt("夜兰喜欢吃什么？")
    print(out)

    out = run_problem_context_prompt_in_character_manner("优菈" ,"夜兰喜欢吃什么？")
    print(out)

    out = run_problem_context_prompt_in_character_manner("魈" ,"璃月有哪些珍奇异兽？")
    print(out)

    out = run_problem_context_prompt_in_character_manner("琳妮特" ,"菲米尼和林尼是什么关系？")
    print(out)

    out = run_problem_context_prompt_in_character_manner("香菱" ,"荻花洲是一个什么样的地方，有哪些传说？")
    print(out)

    out = run_problem_context_prompt_in_character_manner("芙宁娜" ,"那维莱特的古龙大权是什么东西？")
    print(out)

    out = run_problem_context_prompt_in_character_manner("钟离" ,"归终是谁？")
    print(out)

    out = run_problem_context_prompt_in_character_manner("温迪" ,"钟离是一个什么样的人？")
    print(out)

    out = run_problem_context_prompt_in_character_manner("凝光" ,"岩王帝君是一个什么样的人？")
    print(out)

    out = run_problem_context_prompt_in_character_manner("神里绫华" ,"如果想放烟花可以到稻妻找谁？")
    print(out)

    out = run_problem_context_prompt_in_character_manner("宵宫" ,"如果想放烟花可以到稻妻找谁？")
    print(out)

    out = run_problem_context_prompt_in_character_manner("珊瑚宫心海" ,"白夜国的子民遭遇了什么？")
    print(out)

    out = run_problem_context_prompt_in_character_manner("菲谢尔" ,"莫娜在释放元素爆发时会说什么？体现出来她的什么性格特征？")
    print(out)

    out = run_problem_context_prompt_in_character_manner("琴" ,"爱丽丝女士是可莉的妈妈吗？")
    print(out)

    out = run_problem_context_prompt_in_character_manner("艾尔海森" ,"卡维的脾气秉性是什么样的？")
    print(out)
