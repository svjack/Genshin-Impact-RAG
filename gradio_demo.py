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

demo.queue(api_open=False)
demo.launch(max_threads=30, share = True)
