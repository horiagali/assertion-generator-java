from typing import Dict
from langchain_openai import ChatOpenAI
from state import AgentState

llm = ChatOpenAI(
    base_url="http://172.18.96.1:11434/v1",
    api_key="ollama",
    model="qwen3-coder:480b-cloud",
    temperature=0
)

def run_oneshot_logic(state: AgentState) -> Dict:
    prompt = state.get("prompt_context")
    system_msg = (
        "You are an expert Java Test Engineer. "
        "Task: Write high-quality JUnit assertions for maximum mutation coverage. "
        "CRITICAL RULES: "
        "1. DO NOT declare new variables. Use existing variables from the context. "
        "2. DO NOT write 'String result = ...' or similar declarations. "
        "3. ONLY output assertion lines (e.g., assertEquals(expected, actual);). "
        "4. AVOID assertNotNull if more specific assertions are possible. "
        "5. Provide a SERIES of assertions to verify multiple properties. "
        "6. No explanations or markdown."
        "7. NEVER use try-catch blocks or throw checked exceptions (like IOException) unless they are in the target signature. "
    )
    try:
        response = llm.invoke([("system", system_msg), ("human", prompt)])
        prediction = response.content.strip().replace("```java", "").replace("```", "").strip()
        print(f"    >> Generated Assertions:\n{prediction}")
        return {"prediction": prediction}
    except Exception as e:
        print(f"    >> [ERROR] LLM failed: {e}")
        return {"prediction": None}

