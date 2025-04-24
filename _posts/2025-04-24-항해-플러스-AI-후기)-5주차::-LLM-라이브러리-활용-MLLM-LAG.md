---
title: 항해 플러스 AI 후기) 5주차 LLM 라이브러리 활용 - MLLM / RAG
description: >-
  항해 플러스 AI 과정을 진행하면서 5주차에 학습한 내용을 정리한다.
author: 김가은
date: 2025-04-24 20:00:00 +0900
categories: [항해플러스AI]
tags: [항해99, 항해 플러스 AI 후기, AI 개발자, LLM, 딥러닝, 머신러닝, 인공지능, instruction tuning, MLLM, RAG, 자연어 처리]
pin: true
---

이번 주에는 LLM을 좀 더 잘 활용하기 위해 instruction-tuning, MLLM, RAG에 대해 학습하였다.
특히 RAG는 이전에도 들어본 적이 있지만 잘 알지 못하는 개념이었는데, 굉장히 흥미로웠다.

이외에도 멘토링을 진행하면서 LangChain을 좀 더 공부해보면 좋을 것 같아서 찾아보고 있다.
LangChain은 LLM을 잘 활용하기 위한 다양한 기능이 모여있는 라이브러리로, 잘 활용하면 코드를 간결하게 작성할 수 있을 것 같다.

이번 주 심화과제에서는 프로젝트를 본격적으로 시작하기에 앞서 아주 간단한 프로토타입을 만들어보았다.
심화과제를 진행하면서 프로젝트에 대해 고민한 내용들도 정리하였다.

## 5주차 학습한 내용

### 1. Instruction-tuning

LLM의 텍스트 생성 능력만을 가지고 사용자의 질문에 답을 올바른 답을 찾아내는 것은 한계가 있다.
실제로 초창기 LLM들은 질문과 관련된 text를 출력하는 오류가 많았다.

instruction-tuning을 통해 이를 보완할 수 있다.

#### Supervised Fine-tuning(SFT)
instruction-tuning에서는 질문과 답변의 쌍으로 이루어진 data를 사용한다.

```js
  {
    "instruction": "다음 문장을 영어로 번역해주세요: '오늘 날씨가 좋네요'",
    "output": "The weather is nice today."
  },
  {
    "instruction": "파이썬에서 리스트를 정렬하는 방법을 설명해주세요.",
    "output": "파이썬에서 리스트를 정렬하는 방법은 크게 두 가지가 있습니다:\n1. sort() 메서드 사용: list.sort()\n2. sorted() 함수 사용: sorted(list)\nsort()는 원본 리스트를 직접 수정하고, sorted()는 정렬된 새로운 리스트를 반환합니다."
  },
  {
    "instruction": "인공지능이란 무엇인지 초등학생이 이해할 수 있게 설명해주세요.",
    "output": "인공지능은 컴퓨터가 사람처럼 생각하고 배우는 능력이에요. 마치 우리가 경험을 통해 배우는 것처럼, 컴퓨터도 많은 정보를 보고 배워서 문제를 해결할 수 있어요. 예를 들면, 스마트폰에서 사진을 찍으면 자동으로 얼굴을 인식하는 것도 인공지능이 하는 일이랍니다!"
  },
```

supervised fine-tuning(SFT)은 질문은 제외하고 답변에 대해서만 next token prediction을 하는 fine-tuning 방식을 말한다.

SFTTrainer를 사용하여 구현할 수 있다.
```python
from trl import SFTTrainer
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# 모델과 토크나이저 초기화
model = AutoModelForCausalLM.from_pretrained("모델명")
tokenizer = AutoTokenizer.from_pretrained("토크나이저명")

# 트레이너 초기화
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512
)

# 데이터셋 예시
dataset = Dataset.from_list([
    {
        "instruction": "질문 내용",
        "output": "답변 내용"
    }
])

# 데이터 포맷팅
def formatting_prompts_func(example):
    return f"### 질문:\n{example['instruction']}\n\n### 답변:\n{example['output']}"

# 트레이닝 실행
trainer.train()

# 모델 저장
trainer.save_model("저장할_경로")
```

#### instruction-tuning data 생성하기

**Self-Instruct**
prompting과 GPT만을 사용하여 데이터를 만든다.

1. Seed task 생성하기: 생성에 사용할 예시 instruction-tuning data를 사람이 직접 만들고, task pool에 추가한다. seed task는 데이터셋의 품질을 결정하는 중요한 역할을 한다.
2. LLM으로 instruction 생성하기: task pool에서 instruction을 랜덤 샘플링하고, prompting을 통해 LLM으로 새로운 instruction을 생성한다.
```
[Task Pool] -> [랜덤 샘플링] -> [LLM에게 프롬프팅] -> [새로운 Instruction 생성]

예시 프롬프트:
"다음은 AI 모델에게 주어질 수 있는 다양한 지시사항들입니다:
1. '이 문장을 영어로 번역해주세요'
2. '다음 텍스트의 감정을 분석해주세요'

위 예시들과 비슷하지만 다른 새로운 지시사항을 3개 만들어주세요."
```
3. instruction-tuning data 생성하기: 생성된 instruction을 가지고 prompting을 통해 instruction-tuning data를 생성한다.
```
[생성된 Instruction] -> [LLM에게 프롬프팅] -> [응답 생성] -> [Instruction-Output 쌍 생성]

예시:
Instruction: "다음 문장의 핵심 주제를 추출해주세요"
Output: "이 문장에서 다루고 있는 가장 중요한 주제나 개념을 간단히 설명하겠습니다..."
```
4. filtering 후 반복: filtering 후 새로 만든 instruction과 data를 task pool에 추가하고 반복한다. filtering을 통해 지시사항이 명확한지, 응답이 적절한지, 중복되지는 않는지 체크할 수 있다.

**GSM-Plus**
GSM-Plus는 GSM8K라는 수학 데이터셋을 GPT로 변형해서 만든 data이다.
원본 문제의 구조는 유지하면서 숫자를 바꾸거나 문제의 용어들을 바꾸어 데이터를 확장하였다.

**UltraChat**
두 개의 LLM에 특정 seed question들로 대화하게 하여, 이 대화 내역을 instruction-tuning data로 사용한다.
```js
[
    "인공지능의 윤리적 문제에 대해 토론해보겠습니까?",
    "기후 변화에 대처하는 방법을 논의해볼까요?",
    "프로그래밍 언어들의 장단점을 비교해볼 수 있을까요?",
    "미래의 교육 시스템은 어떤 모습일까요?",
    "인간과 AI의 협력 방안에 대해 이야기해보시죠."
]
```
성능이 아직 좋지 않은 open LLM은 closed LLM의 data로 학습시켜 성능을 높일 수 있다.


### 2. Multi-modal Large Language Model(MLLM)
text뿐만 아니라 이미지, 오디오 등 다양한 형태의 입력도 처리할 수 있는 모델이다.

#### Vision-Language Model(VLM)
VLM은 이미지와 텍스트를 모두 입력으로 받아 처리하는 모델이다.

**Vision Transformer(ViT)**
ViT는 이미지를 patch로 분할하여 텍스트처럼 토큰화하고, 이 토큰들을 입력으로 사용하는 모델이다.

Transformer이기 때문에 inductive bias가 적고, text를 입력으로 받는 Transformer와 연계하기 쉽다.

- **Inductive bias**
모델이 학습 과정에서 가지는 사전 가정이나 편향을 말한다.
CNN은 이미지의 중요한 특징은 서로 가까이 있는 픽셀들의 관계에서 발견된다고 가정(지역성)하고, 레이어에 따라 계측정 구조를 갖기 때문에 instructive bias가 높다.
Transformer는 모든 요소들 간의 관계를 동시에 고려하기 때문에 inductive bias가 적다.
데이터가 적으면 CNN이, 데이터가 많으면 ViT가 더 좋다.

#### Vision Language Model(VLM)
VLM(Vision-Language Model)은 Vision(시각)과 Language(언어) 모델을 결합한 멀티모달 모델이다.

ViT를 vision encoder로 사용하여 이미지로부터 token sequence를 생성한다.
projection layer를 사용하여 언어 모델에 맞는 임베딩 크기로 변환한다. 
여기에 text와 결합하여 language model에 입력으로 사용한다.

대표적인 VLM인 LLaVa은 오픈소스로 제공되며, 상대적으로 가벼운 모델 크기에도 높은 성능을 보여준다.

#### GPT API with Image Input
GPT API에 image를 입력으로 사용할 때는 다음과 같은 방법을 사용할 수 있다.

**1. 이미지 파일 사용**

```python
def encode_image_to_base64(image_path):
    # 이미지를 base64로 인코딩
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# 이미지 경로
image_path = "path/to/your/image.jpg"
base64_image = encode_image_to_base64(image_path)

# API 요청 메시지 구성
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "이 이미지에 대해 설명해주세요."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ]
    }
]
```

**2. 이미지 URL 사용**

```python
response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "이 이미지의 주요 객체들을 나열해주세요."
                },
                {
                    "type": "image_url",
                    "image_url": "https://example.com/image.jpg"
                }
            ]
        }
    ],
    max_tokens=300
)
```

### 3. Retrieval Augmented Generation(RAG)

RAG는 모델이 외부 knowledge source에서 정보를 검색하여 추가적인 컨텍스트를 제공하는 방법이다.
knowledge source는 pdf 등 문서, 인터넷 자료, 데이터베이스, 전통적인 검색 엔진 등이 될 수 있다.
LLM에서 나타나는 hallucination을 줄이고, 모델의 응답을 더 정확하게 만들 수 있다.

1. 사용자의 입력을 받는다.
2. 사용자의 입력을 기반으로 외부 데이터베이스에서 유사한 text들을 추출한다.
3. 추출한 text와 사용자의 질문을 합쳐서 prompt를 만든다.
4. prompt를 모델에 입력으로 사용하여 응답을 생성한다.

**LangChain**
LangChain은 LLM을 사용하면서 생기는 문제들을 해결하기 위해 나온 LLM 통합 library이다.
RAG도 LangChain을 통해 구현할 수 있다.

- Runnable component: langchain 객체
- invoked: runnable component를 실행하는 것
- LangChain expression language(LCEL): 각각의 runnable component를 연결하여 파이프라인처럼 사용할 수 있게 해주는 문법


**Knowledge source에서 text 추출하기**
RAG를 사용하기 위해 knowledge source에서 관련된 text를 추출할 때는 다음과 같은 방법을 사용한다.

```
[Knowledge Source] -> [텍스트 추출] -> [임베딩 모델] -> [벡터 DB 저장]
예: PDF 문서 -> 텍스트 청크 -> 임베딩 벡터 -> Pinecone/Faiss 등에 저장
```

임베딩할 때 적절한 chuck size를 사용해야 한다.
한문장은 맥락이 너무 작고, document 전체는 성능이 안좋다.
Splitter를 사용하여 chuck size나 chunk overlap을 조절하여 적절한 크기로 임베딩되었는지 테스트할 수 있다.

rlm/rag-prompt와 같은 라이브러리를 사용하여 RAG에 사용할 최종 프롬프트를 작성할 수 있다.

## 심화과제 회고

이번 심화과제 주제는 프로젝트에 시작하기에 앞서 RAG를 사용해 간단한 프로토타입을 만드는 것이었다.

몇 가지 예시 주제가 있었지만, 프로젝트 주제로 테스트해보고 싶은 주제가 있었다.
과제를 진행하면서 RAG를 사용한 LLM이 어떤 구조를 갖게되는지 알 수 있었고, LangChain도 사용해볼 수 있었다.

이번 프로젝트에서는 **도서 추천 서비스**를 만들어보고 싶었다.
그런데 추천 서비스를 만들기 위해서 충분한 데이터를 확보하는 것이 어려웠다.

우선 공공 API로 제공되는 [국립중앙도서관 사서추천도서목록](https://www.nl.go.kr/NL/contents/N31101030900.do)를 사용하였다.
xml 데이터라서 json으로 변환하였다.

python으로 데이터 전처리를 해본적이 거의 없어서, 데이터가 원하는데로 변환이 되지 않아서 어려웠다.

최종적으로 다음과 같은 데이터 1388개를 확보하였다.

<img width="651" alt="스크린샷 2025-04-24 오후 9 09 30" src="https://github.com/user-attachments/assets/30970baf-d186-4c7a-a8f1-bd0065363da8" />


그리고 RAG를 사용하여 벡터DB에 저장하였다.

프롬프트에 모든 데이터를 넣을 수 없기 때문에 백터DB에서 질문과 유사한 데이터 top3를 뽑아서 프롬프트에 포함시켰다.

openai의 gpt-4o-mini 모델과 open LLM 모델을 모두 사용해보았다.

프롬프트에 top3 중에 가장 추천하는 도서를 선정하고, 그 이유를 같이 답변해달라고 했다.

gpt-4o-mini는 각 도서에 대한 설명이나 추천하는 이유에 대해서 잘 설명해주었지만,
openLLM 모델은 도서를 설명할 때 프롬프트에 들어간 도서 정보에 있는 문장을 그대로 사용하는 경우가 많았고, 추천하는 이유도 어색했다.
또, 응답도 오래걸렸다.

gpt-4o-mini는 답변 자체는 좋았지만, top3 안에서 선정하다보니 다양한 답변이 나오기 어려웠다.

추천 서비스를 만든다면 데이터를 더 확보하는 것이 중요할 것 같다.

그리고 도서 추천 서비스라고 할 때 사용자가 단순히 특정 주제를 물어보는 것보다 현재 기분 상태에 따라 추천해주고 싶었다.

몇 가지 프롬프트로 테스트를 해보았는데 gpt api를 사용하면 그럴듯한 대답을 만들어낼 수 있을 것 같지만 openLLM으로 구현하려면 난이도가 매우 높아질 것 같다

과제를 진행하면서 추천 서비스 자체가 데이터가 중요하다고 느껴서, 프로젝트 주제로 가져갈지는 좀 더 고민이 필요할 것 같다.
