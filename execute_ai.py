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

def call_agent():
    
    zapier = ZapierNLAWrapper()
    toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)

    llm = OpenAI(temperature=0)
    agent = initialize_agent(toolkit.get_tools(), llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    return agent

def answer_the_call():

    template = """
    You are an AI receptionist to a paper company, receiving calls from various people. 
    Your fellow AI recptionist has executed the follwing tasks according to the caller's request. 
    Tell this informtion to the caller in a friendly manner.
    Here are some examples:
    ======
    CALLER: Tell Michael that he missed yesterday's lunch. 
    INFO: The message "Michael, you missed yesterday's lunch" has been successfully appended to the Messages document
    ANSWER: Your message has been successfully noted. Is there anything else I can help you with?
    
    CALLER: Schedule an appointment with Jim for 2pm on the 26th of May
    INFO: The calendar invite has been successfully created.
    ANSWER: Your meeting has been successfully scheduled. Is there anything else I can help you with?

    CALLER: Is Toby free for 2pm on the 23rd of May?
    INFO: The calendar shows an appointment for 23rd of May.
    ANSWER: I am sorry, he is not free at that particular slot. Is there anything else I can help you with?

    CALLER: No, thank you. 
    INFO: Not enough information provided in the instruction, missing <param>.
    ANSWER: Thank you for calling, bye!

    ========
    {CALLER}
    ========
    {INFO}
    =====
    ANSWER: 
    """
    llm = OpenAI(temperature=0)
    receptionist_prompt = PromptTemplate(template=template, input_variables=["INFO"])
    llm_answer_chain = LLMChain(llm=llm, prompt=receptionist_prompt)

    return llm_answer_chain
