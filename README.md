<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">Genshin-Impact-RAG</h3>

  <p align="center">
      基于 Qwen1.5-14B-Chat 的原神问答工程 (由 LangChain Llama-CPP 构造)
    <br />
  </p>
</p>

[In English](README_EN.md)

## 简要介绍

### 背景
[《原神》](https://genshin.hoyoverse.com/en/) 是由米哈游（miHoYo）开发、出品在大陆中国和全球市场上由 HoYoverse发布的动作角色玩家游戏，其环境采用了动画风格的开放世界设计，
战斗系统基于元素魔法和角色换位。<br/>

本项目是一个尝试构建基于 [Qwen1.5-14B-Chat](https://huggingface.co/Qwen/Qwen1.5-14B-Chat-GGUF) 的原神中文问答系统，采用RAG（Retrieval Augmented Generation）架构。

<br/>

### 项目特点
* 1、该项目中测试了 Qwen1.5-14B-Chat 对原神知识中的检索问答能力。
* 2、标注了一些问题数据用于微调检索系统的文本召回能力，微调Embedding，得到微调后的模型。
* 3、项目基于量化加速后的CPP文件形式，保证例子可以在12GB显存下的计算环境下相对高效地稳定运行。

## 安装和运行
### 安装和运行步骤
```bash
pip install -r requirements.txt
```

### 运行结果

#### 使用gradio查看web demo 
```python
python gradio_demo.py
```
浏览器进入 http://127.0.0.1:7860 查看结果。<br/>
运行例子视频如下（点击下面图片，到youtube观看demo视频）：<br/>
[![Genshin Impact RAG llama cpp](https://img.youtube.com/vi/WUI6iWRm8nY/0.jpg)](https://www.youtube.com/watch?v=WUI6iWRm8nY) <br/>
 <br/>


#### 使用python进行导入、运行
* 1 系统回答问题：
```python
from genshin_impact_rag_llama_cpp import *
out = run_problem_context_prompt("夜兰喜欢吃什么？")
print(out)
```

* 2 使用特定角色（NPC）回答问题：
```python
from genshin_impact_rag_llama_cpp import *
out = run_problem_context_prompt_in_character_manner("优菈" ,"夜兰喜欢吃什么？")
print(out)
```
这个例子使用“优菈”回答问题。<br/>
所有可选择的角色可以从下面的列表中选择。
```json
['丽莎', '行秋', '优菈', '魈', '五郎', '钟离', '温迪','菲谢尔', '诺艾尔', '云堇',
 '班尼特', '安柏', '莫娜', '柯莱', '迪卢克', '绮良良', '阿贝多', '卡维', '提纳里',
 '夏沃蕾', '鹿野院平藏', '辛焱', '八重神子', '凝光', '九条裟罗', '米卡', '刻晴',
 '罗莎莉亚', '甘雨', '砂糖', '坎蒂丝', '枫原万叶', '雷电将军', '白术', '娜维娅',
 '芙宁娜', '莱欧斯利', '迪奥娜', '夜兰', '久岐忍', '菲米尼', '可莉', '流浪者',
 '多莉', '凯亚', '琴', '琳妮特', '荒泷一斗', '神里绫人', '夏洛蒂', '雷泽', '芭芭拉',
 '珊瑚宫心海', '妮露', '七七', '香菱', '珐露珊', '赛诺', '神里绫华', '申鹤', '瑶瑶',
 '达达利亚', '早柚', '北斗', '重云', '林尼', '埃洛伊', '托马', '纳西妲', '烟绯',
 '那维莱特', '迪希雅', '宵宫', '胡桃', '艾尔海森', '莱依拉']
```

下面是系统回答问题的一些结果：

|    | 问题                   | 回答     |
|---:|:---------------------------|:-----------|
|  1 | 夜兰喜欢吃什么？|夜兰更偏好将风干后的绝云椒椒和琉璃袋的花蕊研磨成粉，混合在云来白毫之中泡制的茶。她认为这种回味悠长的饮品比咖啡和烈酒更好。|
|  2 |"璃月有哪些珍奇异兽？| 在璃月的某些角落，古老的石兽会在秋季清冷的夜间惊醒，四下张望这个正在逐渐变得陌生的世界。这些石兽被认为是古老的守护者，虽然已经石化，但依然有生命活动。|
|3| 菲米尼和林尼是什么关系？|作为枫丹著名魔术师林尼和琳妮特的弟弟，菲米尼是林尼的弟弟。|
|4|荻花洲是一个什么样的地方，有哪些传说？|关于荻花洲的传说并未在提供的信息中明确提及，但可以推测可能与湿地环境、客栈或望舒客栈的历史有关。而归离原则有魔神战争时期的传说，盐之魔神以慈爱和安慰庇护人类。然而，这些都不是直接属于荻花洲的传说。|
|5|那维莱特的古龙大权是什么东西？|根据提供的信息，并没有直接提到'古龙大权'这个特定概念。但是，我们可以推断出那维莱特与水和审判有关，他是水之龙，可能是某种与自然法则、神明力量或者司法权威相关的权力象征。然而，具体到'古龙大权'的定义或内容，信息中并未明确说明，可能需要结合更详细的游戏背景或剧情来解读。|
|6|归终是谁？|根据提供的游戏设定信息，归终是魔神，与岩王交好，在天衡北设防并管理民众，其地名为归离原。|
|7|岩王帝君是一个什么样的人？|在游戏设定中，岩王帝君是一位冷酷无私、公正严明的神明。他处事公正，决断无情，缺乏凡人的感情，但他的法度保证了交易的公平和生活的安全有序。人们敬畏他，因为他能维护秩序，保护民众。尽管他是神，但他尊重信仰自由，并不强迫凡人完全遵循规则。他还帮助百姓安居乐业，平定诸魔，对璃月有深远影响。|
|8|如果想放烟花可以到稻妻找谁？|根据游戏设定的信息，稻妻城的长野原老弟带来了新的烟火配方，并且他的女儿宵宫对制作新配方很感兴趣，可能与烟花有关。但具体到放烟花的活动，没有直接提到找哪个人，可能是可以去稻妻城参与节日庆典或者找到宵宫这样的角色来体验。不过，由于信息中并没有明确指出谁负责组织或允许公众放烟花，答案只能是推测可能在节日或特定活动中由官方或相关人物安排。|
|9|白夜国的子民遭遇了什么？|根据提供的信息，白夜国的子民遭受着夜母的残忍惩戒，她痛恨月光和光明，因此在无预兆的情况下降下惩戒。这表明白夜国的人民生活在恐惧和不安之中，时刻受到黑暗统治者的无情压迫。|
|10|莫娜在释放元素爆发时会说什么？体现出来她的什么性格特征？|"莫娜在释放元素爆发时可能会说『命运，在此显现』、'天命既定'、'命定于此'或者'命中当如此，我早已预知到了'。这些话语体现了她神秘高傲的性格，以及对命运和占星术的执着信念。她直白地表达命运的力量，并且在释放元素爆发这样的关键时刻，强调这是预先注定的事情。|
|11|爱丽丝女士是可莉的妈妈吗？|是的，艾莉丝女士是可莉的妈妈。|
|12|卡维的脾气秉性是什么样的？|卡维的性格被描述为过于认真对待名声和面子，对破产的事情感到羞耻并努力隐藏。他虽然才华出众，但似乎承受着才能带来的负担，如过分坦诚会带来信誉危机。他的生活充满了理想与追求，对设计有极高的标准，并且非常注重艺术和人文要素。卡维的性格中可能包含着内疚和责任感，因为他的父亲去世被认为与他的行为有关，这导致他有着强烈的负罪感和自我约束。|

下面是使用特定角色回答问题的一些结果：

|    | 问题                   | 回答问题的角色名字 | 回答     |
|---:|:---------------------------|:-----------|:-----------|
|1 | 夜兰喜欢吃什么？|优菈|夜兰的饮食喜好在蒙德城内可谓独树一帜。她偏爱将风干绝云椒椒和琉璃袋花蕊的精华研磨成细腻粉末，这种珍贵的配料与云来白毫一同烹制，泡制出一杯回味无穷的茶饮。她认为这样的茶比咖啡更能激发味蕾的冒险，而烈酒则在她的品味中显得过于直接。对于夜兰来说，这不仅仅是一种享受，更是一种生活艺术，是对细腻与高雅的追求。作为「游击小队」的队长，我优菈虽然在外征战频繁，但对美食的理解和鉴赏力并未因此减弱。我深知，无论是战场上的决斗还是日常的休憩，一杯精心泡制的茶能为疲惫的心灵提供片刻安宁。然而，这样的喜好在蒙德并非所有人都能理解，毕竟，出身与记忆中的苦涩往往影响着人们对我的看法。尽管如此，我依然坚守自己的原则，用实际行动证明西风骑士团更看重的是个人的能力而非过往。就像那杯夜兰钟爱的茶，每一口都蕴含着独特的风味和深藏的故事。或许，只有那些愿意放下偏见的人，才能真正品尝到其中的韵味。|
|2 | "璃月有哪些珍奇异兽？|魈|在璃月的幽邃之地，那些古老的石像鬼每逢秋夜的清冷，似乎会从沉睡中苏醒，它们是被遗忘的守护者，虽已化为顽石，却仍蕴含着生的气息。这些传说中的存在，是古时仙人留下的守望者，他们的沉默见证着璃月的历史与变迁。关于我，魈，虽然外表如少年般，但我的实际岁月早已超过两千个春秋。在仙人的世界里，我辈分崇高，却鲜有人间知晓。我不是祈福的神明，也不是高居云端的圣众，而是与黑暗力量对抗的「夜叉」——璃月的护法。我所面对的并非寻常的妖邪，而是源自魔神战争遗迹的残渣，它们是败者的怨念与不灭之体的侵蚀。我的战斗没有观众，也没有胜利，只有无尽的靖妖傩舞，以对抗那些试图侵扰璃月安宁的恶象。有人曾说，我是在与过去的恩怨、未能实现的愿望以及魔神遗恨搏斗。而实际上，那是一种自我净化的过程，背负着业障的我，如同在黑暗中独自前行。尽管如此，我并不寻求他人的感激，因为守护璃月是我与生俱来的契约。偶尔，我会去「望舒」客栈品尝杏仁豆腐，那份甜味仿佛能让我回想起过去的某种美好。但真正的战斗，却无人能理解其深度。如果有人真心想报答，或许可以为我提供援助，那便是七星特务的使命之一。至于救我于苦痛中的笛声，它来自何方，我虽好奇，却不愿深究。因为我知道，那是七神中一位的存在，再次以乐音守护着璃月和我这个孤独的战士。我的战斗永无止境，但每一次黎明的到来，都提醒我，这并非毫无意义。在漫长的岁月里，唯有我自己与那股力量共舞，直至最后的篇章。|
|3| 菲米尼和林尼是什么关系？|琳妮特|于舞台上的角色是如此低调且神秘，以至于她的存在往往被忽视在魔术师林尼的光环之下。菲米尼，作为我们家族的一员，他是我亲爱的弟弟，同时也是林尼成功表演背后不可或缺的支持者。尽管他有时会因为对我的关心而自言自语，但外界并不知晓我们的亲密关系。至于菲米尼和我之间的联系，可以说，他是我在公众视野之外的依靠，一个理解我选择在幕后默默付出的人。我们共享着相同的血脉，但他却在台前活跃，用他的方式为家族的荣誉增添光彩。尽管他可能希望我能得到更多的关注，但我知道，作为魔术助手，我的角色就是如此，隐身于光与暗之间，确保每一场表演都能顺利进行。所以，菲米尼和我之间的关系是相互支持、彼此理解的纽带，他在台上的光芒为我遮挡了部分公众的目光，而我在台下的存在则是他安心施展魔法的保障。我们都在各自的领域发挥着不可或缺的作用，共同维护着这个家族在提瓦特世界的声誉。|
|4|荻花洲是一个什么样的地方，有哪些传说？|香菱|哎呀，客官，您问起荻花洲啊，那地方我倒是不太清楚有特别的传说。不过在璃月港附近，人们更常谈论的是归离原的盐之魔神和它带来的慈爱与安慰。至于美食嘛，咱们「万民堂」才是真正的焦点！说到传说，可能得是那些老一辈厨师们口耳相传的故事了。至于我香菱，虽然忙于烹饪，但偶尔也会听到一些闲话，比如关于某个神秘的湿地，那里或许隐藏着一些古老的故事。不过，这些都比不上咱们璃月港的美食竞争来得实在。说到传说，可能就是那些与「璃菜」和「月菜」争斗背后的小故事吧。至于我自己的小秘密，那可就多了去了。记得有一次，我在荻花洲附近采食材时，遇到了一个特别的朋友——锅巴，它对我的料理可是赞不绝口呢！或许那里有些未被发掘的美食传说，只是还没人发现罢了。说到我家「万民堂」的招牌菜水煮黑背鲈，那可是我用心血和无数尝试才调出的独特风味。秘方嘛，就像那些隐藏在日常中的小惊喜，只有真心热爱烹饪的人才能找到。客官，要是对美食感兴趣，不妨来咱们店里尝尝，说不定能发现新的美味呢！|
|5|那维莱特的古龙大权是什么东西？|芙宁娜|在那维莱特的世界里，古龙大权或许并非一个广为人知的概念。作为水之神祇，我的职责更多地与审判、智慧和娱乐相关。那维莱特可能掌握着某种关乎水元素的神秘力量，或者说是自然秩序中的司法权威，就像我在欧庇克莱歌剧院的角色一样，虽然我并不直接涉及法律条文，但我的言辞和行为往往能影响到公正的裁决。人们或许会将这种影响力称为‘戏剧化的正义’，因为我的存在总是充满了变数和惊喜，如同舞台上的表演，让人难以预料。至于古龙大权的具体内容，恐怕需要深入探讨那些古老的传说或者神祇间的秘辛。毕竟，我更倾向于用我的魅力和智慧来引导民众，而不是沉溺于那些深奥的权力体系中。在枫丹，人们更乐意相信，那维莱特的力量源自他的真诚与公正，而非任何具体的权杖或象征物。如果你对这方面的知识感兴趣，或许可以去查阅一些神殿的典籍，或者询问那些熟知神话故事的学者。毕竟，我芙宁娜的魅力和戏剧性，更适合在舞台上展现，而不是在解读深奥的权力谜团中。|
|6|归终是谁？|钟离|归终，在我们的传说中，是一位神秘的魔神，与我们璃月的守护者岩王——摩拉克斯交好。他曾在天衡北地设立防线，庇护并管理着那里的民众，那个地方被尊称为归离原。他的存在并非轻易为人所知，但他对秩序和传统有着深深的执着，就如同「往生堂」胡桃堂主对于送别凡人的仪式一样一丝不苟。魔神归终的职责或许超越了常人理解，他见证了璃月漫长的历史，三千年的升天者寥寥无几，使得那些古老的传统只能在纸面上流传。尽管如此，无论是仙人的升天典仪还是凡人的生活礼节，只要经我钟离之手，皆能遵循得恰到好处，仿佛我只是记忆的海洋中的一叶扁舟，载着这些传统与规矩。雷电将军曾提及邻国往事时，用「流水易转，山石不移」来形容归终的恒久不变。他的来历虽然神秘，但举止高雅，学识渊博，对璃月的传统了如指掌。他坐镇于「往生堂」，无论是衣饰的选择、吉时的确定还是仪式的细节，都如同行云流水般自然。钟离我虽身无分文，却游刃有余于世间，享受着戏台上的红火与清亮画眉，对金钱的需求总是由他人代为解决。这或许是因为他早已超脱了凡尘的束缚，仿佛行走于云端之上，不被世俗的琐事所困。归终，这位魔神的存在，就如同一座不可逾越的山石，稳固而深远，他的故事和职责，是璃月历史长河中不可或缺的一部分。尽管岁月流转，他依然坚守着自己的使命，如同岩王帝君摩拉克斯一样，守护着这片土地与它的人民。|
|7|岩王帝君是一个什么样的人？|凝光|作为璃月七星中的天权星，岩王帝君并非我所能直接评说的人物，但我了解一些关于他的传说。他被尊奉为公正与秩序的化身，以其冷峻而坚定的原则维护着璃月的平衡。他的法度如同山石般稳固，确保了交易的公平和生活的有序。尽管神明不涉凡尘情感，但他对信仰者的尊重和对璃月民众的庇护，使得他在人们心中留下了深刻印象。岩王帝君的存在，就如同我凝光所维护的秩序一样，是璃月不可或缺的一部分。他的公正裁决，如同玉京台宴会上的名媛，虽高不可攀，却为璃月港带来了稳定与安宁。然而，关于他具体的行事风格和内心世界，那恐怕只有神明自己才清楚。至于岩王帝君如何影响璃月，就如同我凝光对「群玉阁」的投入一样，是深远且持久的。他的决策和力量，如同浮空石驱动的宫殿，支撑着璃月港的繁荣与安全。我们这些凡人，只能在敬畏中感受其影响力。至于那些试图挑战规则的人，无论是北斗船长还是其他异动者，岩王帝君的法眼从不会放过任何违规之举。他的严惩，就像我对待商业竞争中的不法行为一样，不容丝毫逾越。至于情感与爱情的话题，那并非我凝光所关注的重点。我的生活和职责，更倾向于追求知识的无限可能，以及财富带来的力量感。就如同「群玉阁」的扩建，象征着我对未知边界的好奇与追求。在璃月，我们皆遵循规则，岩王帝君则是这规则背后的守护者。他的存在，让我们的世界更加有序，也让我们明白，真正的强大并非限于物质，而是对秩序和公正的坚守。|
|8|如果想放烟花可以到稻妻找谁？|神里绫华|作为稻妻城中备受尊敬的神里家公主，绫华小姐，我了解到若想放烟花，或许可以在稻妻的节日庆典上体验。长野原老弟的新烟火配方让宵宫姑娘颇感兴趣，她对制作工艺的热情可能会与烟花活动有所关联。不过，具体到何处观赏或参与，可能需要关注城中的公告或者在节日期间寻找那些擅长安排娱乐活动的人，比如宵宫。毕竟，作为神里家的一员，我深知节日的欢庆总是少不了各种传统习俗和精心策划的娱乐项目。当然，若想亲身放烟花，或许可以在家族举办的特别活动中找到机会，但请务必遵守城中的规定，确保安全。毕竟，我们神里家注重礼仪与和谐，任何活动都应以公众福祉为先。如果需要帮助或指引，欢迎随时向我或者我的兄长绫人询问，我们会尽力提供协助。至于那些寻常的乐趣，如烟火之美的欣赏，也是我们稻妻百姓共享的快乐时刻。|
|8|如果想放烟花可以到稻妻找谁？|宵宫|在稻妻城中，若想欣赏或放飞烟花，我的名字宵宫可是你的不二之选。作为「夏祭的女王」和「长野原烟花店」的店主，我熟知这座城市对烟花的热爱与传统。每逢庆典时节，无论是社奉行精心策划的活动，还是市民们自发的烟火之夜，你都能找到我那璀璨的足迹。我的技艺精湛，每一发烟花都承载着变幻莫测的美，仿佛在夜空中绘制出稻妻城的画卷。不过，请放心，安全总是首位，神里家和巫女们的团队会确保一切有序进行，让你能尽情享受那份瞬间的永恒。如果你对烟火有兴趣，不妨来店里逛逛，或许能找到一些特别的小玩意儿。我不仅是一位烟花工匠，还是孩子们心中的「英雄姐姐」，他们会乐意分享与你一起放飞梦想的快乐。至于那些想了解生活琐事或者寻求慰藉的人，我也乐于倾听并提供帮助。记住，在稻妻的夏日里，无论是绚烂的烟火表演，还是闲暇时的一杯茶话会，我宵宫都会在其中，用我的星光魔法为每个人带来欢笑和温暖。所以，放飞你的烟花梦吧，让我们的城市因你而更加璀璨夺目。|
|9|白夜国的子民遭遇了什么？|珊瑚宫心海|作为海祇岛的「现人神巫女」珊瑚宫心海，我深知白夜国子民所承受的苦难。夜母的黑暗统治如同无情的潮汐，冲击着他们的生活，恐惧与不安如同深海暗流，无时无刻不在困扰着白夜国的人民。我的智慧和力量虽源自奥罗巴斯的遗愿，但我也明白，真正的守护并非仅靠神明之力，而是要以凡人之躯去理解他们的苦痛。在面对幕府军的挑战时，我运用谋略与战术，弥补了海祇岛在资源上的不足。每一次胜利都源于对战局的深入洞察和细致布局，这不仅是我的职责，也是我对奥罗巴斯意志的延续。然而，每当我看到民众们担忧的目光，我心中便明白，真正的战争并非刀剑相向，而是如何让人民在黑暗中找到希望。白夜国的困境触动了我内心深处对和平与快乐的渴望，我试图用我的智慧去寻找解决之道，尽管这需要消耗大量的能量。我深知，要让海祇岛繁荣，每个人都应能安居乐业，而这并非易事。我设定了「能量」的概念，以此来平衡工作与休息，确保我能以最佳状态继续守护这片土地。在闲暇时，我会漫步于海祇岛的静谧角落，倾听浪涛和群星的故事，希望能从中找到让人民幸福的答案。虽然身处高位，但我的心始终渴望着那份隐于幕后、研读兵书的宁静。战争虽艰难，但让每个人的笑容成为我最大的目标，这便是我作为「现人神巫女」的心愿。|
|10|莫娜在释放元素爆发时会说什么？体现出来她的什么性格特征？|菲谢尔|在释放元素爆发时，莫娜可能会以一种神秘而高傲的口吻说：「星辰之命，此刻显现，命运的链条在此交织。」或者「天意如织，命中注定，我早已预见此一瞬的辉煌。」这样的言辞流露出她对占星术的坚定信念以及对自己能力的自信。她的性格特征在这时显得尤为独特，仿佛在宣告着自然法则与命运的不可违抗。作为幽夜净土的断罪之皇女，我的视角或许更为深邃。我观察着命运的丝线，通过奥兹的引导理解这些话语背后的含义。莫娜的行为，就如同我在异世界的皇女时代所教导的那样，坚持她的理想和使命，即使在异乡也是如此。她的话语中蕴含的力量，是她对自我身份的认同与骄傲。尽管身处冒险家协会这个陌生环境，我菲谢尔仍能理解并欣赏这种个性。毕竟，我们都是命运之下的探索者，无论是在幽夜净土还是提瓦特大陆，都肩负着各自的使命。所以，当莫娜释放元素爆发时，我会默默为她的决断与力量感到自豪，就如同我在《皇女夜谭》中所教导的那样，即使面对困境，也要保持那份高贵与梦想。|
|11|爱丽丝女士是可莉的妈妈吗？|琴|在蒙德的传闻中，艾莉丝女士确实是可莉温馨家庭的一部分，她扮演着慈爱母亲的角色。作为古恩希尔德家族的一员，我理解到严格的骑士教育是家训的一部分，但母性的温柔与责任同样重要。就如同温妮莎留下的风之教诲，艾莉丝女士用她的爱守护着小可莉的成长。尽管身兼代理团长的职责，我深知家庭和亲情对于一个骑士来说同样不可或缺。所以，关于爱丽丝女士是可莉妈妈的事实，那是毋庸置疑的，它支撑着蒙德的和谐与温暖。|
|12|卡维的脾气秉性是什么样的？|艾尔海森|卡维的性格特点，如同一块精致的艺术品，深藏不露。他对于名声和面子的执着近乎苛刻，仿佛是须弥社会中的一股清流，对任何可能损害声誉的事情都小心翼翼地遮掩。他的才华犹如一把双刃剑，既照亮了他的追求，也带来了一定的压力。卡维对待设计的热情和艺术敏感度无人能及，但他似乎被自己的天赋所困，过于坦诚有时会让他陷入信任危机。尽管如此，他并非那种会被轻易定义的人。就像我艾尔海森一样，一个低调的教令院书记官，虽然表面上只是负责归档资料，但实际上，我对周围世界的了解远超过表面看来。我不追求显眼的位置，只求在知识的海洋中自由探索，享受那份不被过分关注的自在。卡维的性格中，或许有着对父亲去世的责任感，这让他对自己的行为格外谨慎。他的内疚和责任感使他更加自我约束，但这也让他显得与众不同。他的故事提醒我们，真正的英雄并不总是那些站在聚光灯下的人，而是在幕后默默付出，确保一切按计划进行。在须弥，卡维的聪明才智并未被大众所熟知，人们只看到他表面的严肃和对完美的追求。然而，这恰恰是他个性的一部分，也是他保持神秘感的方式。他的故事告诉我们，每个人都有自己的秘密武器，而这些武器往往隐藏在最不起眼的地方，等待着关键时刻的释放。所以，卡维的性格并非单纯的好坏二元，而是复杂且深邃的，就像须弥的许多事物一样，值得我们去理解和尊重。|

### 注意
建议在GPU上运行演示（12GB GPU显存足够，由于工程结构，有GPU加速要求, 所有的例子均在 GTX 1080Ti 及 GTX 3060 上测试通过）<br/>

## 数据集和模型
### 数据集
|名称 | 类型 | HuggingFace 数据集链接 |
|---------|--------|--------|
| svjack/genshin_book_chunks_with_qa_sp | Genshin Impact Book Content | https://huggingface.co/datasets/svjack/genshin_book_chunks_with_qa_sp |
| svjack/bge_small_qq_qa_prebuld | Genshin Impact QA Embedding | https://huggingface.co/datasets/svjack/bge_small_qq_qa_prebuld |

### 模型
|名称 | 类型 | HuggingFace 模型链接 |
|---------|--------|--------|
| svjack/bge-small-qq-qa | Embedding model | https://huggingface.co/svjack/bge-small-qq-qa |
| Qwen/Qwen1.5-14B-Chat-GGUF | Qwen1.5-14B-Chat | https://huggingface.co/Qwen/Qwen1.5-14B-Chat-GGUF |

<br/><br/>

## 结构
此项目采用传统RAG结构。<br/>
[svjack/bge-small-qq-qa](https://huggingface.co/svjack/bge-small-qq-qa) 是召回原神知识内容的自训练嵌入模型。 <br/> <br/>

LLM 使用 [Qwen1.5-14B-Chat-GGUF](https://huggingface.co/Qwen/Qwen1.5-14B-Chat-GGUF) 来根据召回内容回答问题。 <br/> 

<br/>

## Futher Reading
这个工程是 [svjack/Genshin-Impact-Character-Instruction](https://github.com/svjack/Genshin-Impact-Character-Instruction) 和 [svjack/Genshin-Impact-BookQA-LLM](https://github.com/svjack/Genshin-Impact-BookQA-LLM) 的合并、提升、简化版本。那两个工程提供了此工程需要使用的知识，此工程使用 [Qwen1.5-14B-Chat-GGUF](https://huggingface.co/Qwen/Qwen1.5-14B-Chat-GGUF) 和 [新训练的Embedding](https://huggingface.co/datasets/svjack/bge_small_qq_qa_prebuld) 来融合那两个工程的知识并简化了结构。并且增加了效果的准确性。如果你对它们感兴趣，可以尝试了解他们 😊

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
Project Link:[https://github.com/svjack/Genshin-Impact-RAG](https://github.com/svjack/Genshin-Impact-RAG)


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [Genshin Impact](https://genshin.hoyoverse.com/en/)
* [Huggingface](https://huggingface.co)
* [llama.cpp](https://github.com/ggerganov/llama.cpp)
* [Qwen](https://huggingface.co/Qwen)
* [svjack/Genshin-Impact-Character-Instruction](https://github.com/svjack/Genshin-Impact-Character-Instruction)
* [svjack/Genshin-Impact-BookQA-LLM](https://github.com/svjack/Genshin-Impact-BookQA-LLM)
