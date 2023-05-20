from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
from langchain.tools.file_management import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool
import openai
import re
from tempfile import TemporaryDirectory
from langchain.utilities.zapier import ZapierNLAWrapper
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.agents import initialize_agent
import os
from langchain.agents import AgentType
from langchain import PromptTemplate

os.environ["ZAPIER_NLA_API_KEY"] = os.environ.get("ZAPIER_NLA_API_KEY", "")
openai.api_key = ""

def call_agent():
    
    openai.api_key = "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"

    zapier = ZapierNLAWrapper()
    toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)

    llm = OpenAI(temperature=0)
    agent = initialize_agent(toolkit.get_tools(), llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    return agent

def answer_the_call():

    template = """
    You are an AI receptionist to a paper company, receiving calls from various people. 
    Your fellow AI recptionist has executed the follwing tasks. 
    Tell this informtion to the caller in a friendly manner.
    Here are some examples:
    ======
    INFO: The message "Paul, you missed yesterday's lunch" has been successfully appended to the Messages document
    ANSWER:Your message has been successfully noted. Is there anything else I can help you?
    
    INFO: The calendar invite has been successfully created.
    ANSWER: Your meeting has been successfully scheduled. Is there anything else I can help you?
    ========
    {INFO}
    =====
    ANSWER: 
    """
    openai.api_key = ""
    llm = OpenAI(temperature=0)
    receptionist_prompt = PromptTemplate(template=template, input_variables=["INFO"])
    llm_answer_chain = LLMChain(llm=llm, prompt=receptionist_prompt)

    return llm_answer_chain
