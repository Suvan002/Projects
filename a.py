import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import numexpr
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import tool
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic import hub
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain_classic.memory import ConversationBufferMemory
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.callbacks import StreamlitCallbackHandler 

load_dotenv()

@st.cache_resource
def load_agent():
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-1.5B-Instruct", 
        task="conversational",
        max_new_tokens=120,
        temperature=0,
        do_sample=False,
        top_p=0.9,
        repetition_penalty=1.1
    )
    chat_model = ChatHuggingFace(llm=llm)
    
    @tool
    def math_solver(question: str) -> str:
        """Solve any math problem with natural language."""
        try:
            return str(numexpr.evaluate(question))
        except:
            return "Invalid expression"

    search_tool = DuckDuckGoSearchRun()

    @tool  
    def python_exec(code: str) -> str:
        """Run Python for complex calculations. Always print() results."""
        return PythonREPL().run(code)

    tools = [math_solver, search_tool]

    system_prompt = '''AI TUTOR for grades 6-12.

RULES FOR NUMERICAL PROBLEMS:
1. Use math_solver() for arithmetic/algebra
2. Use python_exec() for calculus/physics/geometry  
3. ALWAYS explain steps clearly
4. Ask "What's next?" for easy problems
5. Give hints first → full solution second
6. Tag [TOPIC: algebra] etc.
7. Simple words only, no jargon

Example: "Solve 2x+3=7" → solve_math("solve 2x+3=7") → explain steps.'''

    react_prompt = hub.pull("hwchase17/react")
    prompt = PromptTemplate.from_template(
        react_prompt.template + "\nTutor Rules: " + system_prompt + "\nChat History: {chat_history}\n{input}"
    )

    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        chat_memory=msgs,
        return_messages=True
    )

    agent = create_react_agent(chat_model, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools,
        memory=memory, 
        verbose=False,  
        handle_parsing_errors=True,
        memory_key="chat_history",
        max_iterations=50,
        max_execution_time=100
    )
    return agent_executor, msgs

st.title("AI Tutor Chat")

if "initialized" not in st.session_state:
    st.session_state.initialized = True
    agent_executor, msgs = load_agent()
    if len(msgs.messages) == 0:
        st.session_state.messages = [{"role": "assistant", "content": "Welcome to the AI Tutor! Ask me anything about grades 6-12 math, science, etc."}]
    else:
        st.session_state.messages = [{"role": m.type, "content": m.content} for m in msgs.messages]
else:
    agent_executor, msgs = load_agent()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True) 
        result = agent_executor.invoke(
            {"input": prompt}, 
            config={"callbacks": [st_cb]}
        )
        response = result['output']
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()
