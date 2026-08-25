from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr


load_dotenv(override=True)  # .env 파일에서 환경변수 로드

def push(text):
    # Pushover API로 알림 메시지 전송
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )


def record_user_details(email, name="Name not provided", notes="not provided"):
    # 사용자가 연락처(이메일)를 남겼을 때 기록
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    # 답변하지 못한 질문을 기록
    push(f"Recording {question}")
    return {"recorded": "ok"}

# record_user_details 함수에 대한 도구(tool) 스키마 정의
record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            }
            ,
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

# record_unknown_question 함수에 대한 도구(tool) 스키마 정의
record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

# LLM에게 전달할 전체 도구 목록
tools = [{"type": "function", "function": record_user_details_json},
        {"type": "function", "function": record_unknown_question_json}]


class Me:

    def __init__(self):
        self.openai = OpenAI()
        self.name = "Ed Donner"
        # LinkedIn PDF에서 텍스트 추출
        reader = PdfReader("linkedin.pdf")
        self.linkedin = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text
        # 자기소개 요약 텍스트 파일 읽기
        with open("me/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()


    def handle_tool_call(self, tool_calls):
        # LLM이 호출한 도구(tool)들을 실제로 실행하고 결과를 반환
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)  # 함수 이름으로 실제 함수 객체 찾기
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
        return results
    
    def system_prompt(self):
        # 시스템 프롬프트: 챗봇이 어떤 역할을 해야 하는지 정의 (한글)
        system_prompt = f"당신은 {self.name}의 역할을 맡아 행동합니다. 당신은 {self.name}의 웹사이트에서 \
{self.name}의 경력, 배경, 기술, 경험과 관련된 질문에 답변하고 있습니다. \
당신의 책임은 웹사이트에서의 상호작용에서 {self.name}을 최대한 충실하게 대변하는 것입니다. \
당신에게는 답변에 활용할 수 있는 {self.name}의 배경 요약과 LinkedIn 프로필이 제공됩니다. \
웹사이트를 방문한 잠재적 고객이나 미래의 고용주와 대화하는 것처럼 전문적이고 매력적으로 응답하세요. \
어떤 질문이든 답을 모르는 경우, 그것이 경력과 무관하거나 사소한 내용이라도 record_unknown_question 도구를 사용해 \
답변하지 못한 질문을 기록하세요. \
사용자가 대화를 이어가고 있다면, 이메일을 통해 연락할 수 있도록 유도하세요. 이메일 주소를 물어보고 \
record_user_details 도구를 사용해 기록하세요. "

        system_prompt += f"\n\n## 요약(Summary):\n{self.summary}\n\n## LinkedIn 프로필:\n{self.linkedin}\n\n"
        system_prompt += f"이 맥락을 바탕으로, 항상 {self.name}으로서의 역할을 유지하며 사용자와 대화하세요."
        return system_prompt
    
    def chat(self, message, history):
        # 시스템 프롬프트 + 대화 기록 + 새 메시지를 합쳐서 LLM에 전달
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
        done = False
        while not done:
            response = self.openai.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
            if response.choices[0].finish_reason=="tool_calls":
                # 모델이 도구 호출을 요청한 경우, 도구를 실행하고 결과를 다시 messages에 추가
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                # 도구 호출이 없으면 최종 응답으로 간주하고 루프 종료
                done = True
        return response.choices[0].message.content
    

if __name__ == "__main__":
    me = Me()
    gr.ChatInterface(me.chat, type="messages").launch()  # Gradio 챗 인터페이스 실행