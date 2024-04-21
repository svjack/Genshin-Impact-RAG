<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">Genshin-Impact-RAG</h3>

  <p align="center">
   		A Genshin Impact Question Answer Project supported by Qwen1.5-14B-Chat (build by LangChain Llama-CPP)
    <br />
  </p>
</p>

[中文介绍](README.md)

## Brief introduction

### BackGround
[Genshin Impact](https://genshin.hoyoverse.com/en/) is an action role-playing game developed by miHoYo, published by miHoYo in mainland China and worldwide by Cognosphere, 
HoYoverse. The game features an anime-style open-world environment and an action-based battle system using elemental magic and character-switching. 

This project is an attempt to build Chinese Q&A on Qwen1.5-14B-Chat supported RAG system.

<br/>

### Project Feature
* 1. This project tested the retrieval and question-answering capabilities of Qwen1.5-14B-Chat in the Genshin Impact bibliography.
* 2. Annotate some question data to fine-tune the text recall capability of the retrieval system, fine-tune Embedding, and obtain the fine-tuned model.
* 3. The project is based on the quantitatively accelerated CPP file format to ensure that the example can run relatively efficiently and stably in a computing environment with 12GB of gpu memory.

### Try Demo on the fly

## Installation and Running Results
### Install
```bash
pip install -r requirements.txt
```

### Running Results

Following are results of problem in simple way.

|    | Question                   | Answer     |
|---:|:---------------------------|:-----------|
|  1 | 夜兰喜欢吃什么？|夜兰更偏好将风干后的绝云椒椒和琉璃袋的花蕊研磨成粉，混合在云来白毫之中泡制的茶。她认为这种回味悠长的饮品比咖啡和烈酒更好。|
|  2 |"璃月有哪些珍奇异兽？| 在璃月的某些角落，古老的石兽会在秋季清冷的夜间惊醒，四下张望这个正在逐渐变得陌生的世界。这些石兽被认为是古老的守护者，虽然已经石化，但依然有生命活动。|


Following are results of problem answer by specifical characters.

|    | Question                   | Character Answer this Question| Answer     |
|---:|:---------------------------|:-----------|:-----------|
|  1 | 夜兰喜欢吃什么？|优菈|夜兰的饮食喜好在蒙德城内可谓独树一帜。她偏爱将风干绝云椒椒和琉璃袋花蕊的精华研磨成细腻粉末，这种珍贵的配料与云来白毫一同烹制，泡制出一杯回味无穷的茶饮。她认为这样的茶比咖啡更能激发味蕾的冒险，而烈酒则在她的品味中显得过于直接。对于夜兰来说，这不仅仅是一种享受，更是一种生活艺术，是对细腻与高雅的追求。作为「游击小队」的队长，我优�虽然在外征战频繁，但对美食的理解和鉴赏力并未因此减弱。我深知，无论是战场上的决斗还是日常的休憩，一杯精心泡制的茶能为疲惫的心灵提供片刻安宁。然而，这样的喜好在蒙德并非所有人都能理解，毕竟，出身与记忆中的苦涩往往影响着人们对我的看法。尽管如此，我依然坚守自己的原则，用实际行动证明西风骑士团更看重的是个人的能力而非过往。就像那杯夜兰钟爱的茶，每一口都蕴含着独特的风味和深藏的故事。或许，只有那些愿意放下偏见的人，才能真正品尝到其中的韵味。|
|  2 | "璃月有哪些珍奇异兽？|魈|在璃月的幽邃之地，那些古老的石像鬼每逢秋夜的清冷，似乎会从沉睡中苏醒，它们是被遗忘的守护者，虽已化为顽石，却仍蕴含着生的气息。这些传说中的存在，是古时仙人留下的守望者，他们的沉默见证着璃月的历史与变迁。关于我，魈，虽然外表如少年般，但我的实际岁月早已超过两千个春秋。在仙人的世界里，我辈分崇高，却鲜有人间知晓。我不是祈福的神明，也不是高居云端的圣众，而是与黑暗力量对抗的「夜叉」——璃月的护法。我所面对的并非寻常的妖邪，而是源自魔神战争遗迹的残渣，它们是败者的怨念与不灭之体的侵蚀。我的战斗没有观众，也没有胜利，只有无尽的靖妖傩舞，以对抗那些试图侵扰璃月安宁的恶象。有人曾说，我是在与过去的恩怨、未能实现的愿望以及魔神遗恨搏斗。而实际上，那是一种自我净化的过程，背负着业障的我，如同在黑暗中独自前行。尽管如此，我并不寻求他人的感激，因为守护璃月是我与生俱来的契约。偶尔，我会去「望舒」客栈品尝杏仁豆腐，那份甜味仿佛能让我回想起过去的某种美好。但真正的战斗，却无人能理解其深度。如果有人真心想报答，或许可以为我提供援助，那便是七星特务的使命之一。至于救我于苦痛中的笛声，它来自何方，我虽好奇，却不愿深究。因为我知道，那是七神中一位的存在，再次以乐音守护着璃月和我这个孤独的战士。我的战斗永无止境，但每一次黎明的到来，都提醒我，这并非毫无意义。在漫长的岁月里，唯有我自己与那股力量共舞，直至最后的篇章。|


### Note
I recommand you run the demo on GPU (12GB gpu memory is enough, all examples have tested on single GTX 1080Ti or GTX 3060) <br/><br/>

## Datasets and Models
### Datasets
|Name | Type | HuggingFace Dataset link |
|---------|--------|--------|
| svjack/genshin_book_chunks_with_qa_sp | Genshin Impact Book Content | https://huggingface.co/datasets/svjack/genshin_book_chunks_with_qa_sp |
| svjack/bge_small_book_chunks_prebuld | Genshin Impact Book Embedding | https://huggingface.co/datasets/svjack/bge_small_book_chunks_prebuld |

### Basic Models
|Name | Type | HuggingFace Model link |
|---------|--------|--------|
| svjack/bge-small-book-qa | Embedding model | https://huggingface.co/svjack/bge-small-book-qa |
| svjack/setfit_info_cls | Text Classifier | https://huggingface.co/svjack/setfit_info_cls |

### LLM Models
|Name | Type | HuggingFace Model link |
|---------|--------|--------|
| svjack/chatglm3-6b-bin | ChatGLM3-6B 4bit quantization | https://huggingface.co/svjack/chatglm3-6b-bin |
| svjack/mistral-7b | Mistral-7B 4bit quantization | https://huggingface.co/svjack/mistral-7b |

<br/><br/>

## Architecture
This project has a traditional RAG structure.<br/>
[svjack/bge-small-book-qa](https://huggingface.co/svjack/bge-small-book-qa) is a self-trained embedding model
for recall genshin book contents (split by langChain TextSplitter). [svjack/setfit_info_cls](https://huggingface.co/svjack/setfit_info_cls) is a self-trained text classifier for determine whether the content is relevant to the query. <br/> <br/>

LLM Part have 4 different llm frameworks: [HayStack](https://github.com/deepset-ai/haystack) [chatglm.cpp](https://github.com/li-plus/chatglm.cpp) [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) [ollama](https://github.com/ollama/ollama) one can choose to answer the query based on the content recalled by embedding and filter out by text classifier.<br/> 

### Note
[HayStack](https://github.com/deepset-ai/haystack) [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) and [ollama](https://github.com/ollama/ollama) are repos contains many different llms. You can try to use different llms and change the model name or model 
file in the gradio scripts.<br/> * For the ability of understanding the query and context, i recommand you use Mistral-7B in Huggingface Inference Api or Intel/neural-chat in ollama. <br/> * For the ability of answer quality in Chinese, i recommand you Qwen-7B in ollama or ChatGLM3-6B in chatglm.cpp. 

<br/>

## Futher Reading
I also release a project about Genshin Impact Character Instruction Models tuned by Lora on LLM (build by ChatGLM3-6B-base Chinese-Llama-2-13B), an attempt to give Instruction Model demo for different genshin impact characters (about 75 characters) <br/>
If you are interested in it, take a look at [svjack/Genshin-Impact-Character-Instruction](https://github.com/svjack/Genshin-Impact-Character-Instruction) 😊


<br/>

<!-- CONTACT -->
## Contact

<!--
Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com
-->
svjack - https://huggingface.co/svjack - svjackbt@gmail.com - ehangzhou@outlook.com

<!--
Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)
-->
Project Link:[https://github.com/svjack/Genshin-Impact-BookQA-LLM](https://github.com/svjack/Genshin-Impact-BookQA-LLM)


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [Genshin Impact](https://genshin.hoyoverse.com/en/)
* [Huggingface](https://huggingface.co)
* [Mistral-7B](https://mistral.ai/news/announcing-mistral-7b/)
* [LangChain](https://github.com/langchain-ai/langchain)
* [SetFit](https://github.com/huggingface/setfit)
* [HayStack](https://github.com/deepset-ai/haystack)
* [chatglm.cpp](https://github.com/li-plus/chatglm.cpp)
* [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
* [ollama](https://github.com/ollama/ollama)
* [svjack/Genshin-Impact-Character-Instruction](https://github.com/svjack/Genshin-Impact-Character-Instruction)
* [svjack](https://huggingface.co/svjack)
