import gradio as gr
from genshin_impact_rag_llama_cpp import *

all_characters_in_settings_input = ["系统"] + all_characters_in_settings

with gr.Blocks() as demo:
    gr.Markdown("""<center><font size=8>🐑 Genshin Impact RAG QA by Qwen-1.5-14B-Chat in LLama-CPP Bot 🌲</center>""")

    with gr.Row():
        gpt_name = gr.Dropdown(choices = all_characters_in_settings_input, label = "🤖回答者",
            interactive = True, value = "系统"
        )
        question = gr.Textbox("璃月有哪些绝美的山水？" ,
            label = "❓问题（可编辑）", interactive = True, lines = 2)

    with gr.Row():
        answer = gr.Textbox(label = "📖回答", interactive = True)

    with gr.Row():
        clear_history = gr.Button("🧹 清空历史")
        submit = gr.Button("🚀 发送")

    submit.click(
        lambda a, b: run_problem_context_prompt(b.value if hasattr(b, "value") else b)["给出的答案"] if a == "系统" else \
        run_problem_context_prompt_in_character_manner(a.value if hasattr(a, "value") else a,
                                                       b.value if hasattr(b, "value") else b),
        [gpt_name, question],
        answer
    )
    clear_history.click(lambda _: "", answer, answer)

    gr.Examples(
        [
            ["系统", "璃月有哪些绝美的山水"],
            ["系统", "璃月有哪些绝美的山水，这些山水有多么壮丽？用一段风景描写进行表达。"],
            ["云堇", "璃月有哪些绝美的山水，这些山水有多么壮丽？用一段风景描写进行表达。"],
            ["系统", "夜兰喜欢吃什么？"],
            ["优菈", "夜兰喜欢吃什么？"],
            ["魈", "璃月有哪些珍奇异兽？"],
            ["琳妮特", "菲米尼和林尼是什么关系？"],
            ["香菱", "荻花洲是一个什么样的地方，有哪些传说？"],
            ["芙宁娜", "那维莱特的古龙大权是什么东西？"],
            ["钟离", "归终是谁？"],
            ["温迪", "钟离是一个什么样的人？"],
            ["凝光", "岩王帝君是一个什么样的人？"],
            ["神里绫华", "如果想放烟花可以到稻妻找谁？"],
            ["宵宫", "如果想放烟花可以到稻妻找谁？"],
            ["珊瑚宫心海", "白夜国的子民遭遇了什么？"],
            ["菲谢尔", "莫娜在释放元素爆发时会说什么？体现出来她的什么性格特征？"],
            ["琴", "爱丽丝女士是可莉的妈妈吗？"],
            ["艾尔海森", "卡维的脾气秉性是什么样的？"],
        ],
        [gpt_name, question],
        label = "例子"
    )

demo.queue(api_open=False)
demo.launch(max_threads=30, share = True)
