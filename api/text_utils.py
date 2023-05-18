# coding: utf-8
import re

from langchain.prompts import PromptTemplate

def clean_pdf_text(text: str) -> str:
    """Cleans text extracted from a PDF file."""
    # TODO: Remove References/Bibliography section.
    return remove_citations(text)

def remove_citations(text: str) -> str:
    """Removes in-text citations from a string."""
    # (Author, Year)
    text = re.sub(r'\([A-Za-z0-9,.\s]+\s\d{4}\)', '', text)
    # [1], [2], [3-5], [3, 33, 49, 51]
    text = re.sub(r'\[[0-9,-]+(,\s[0-9,-]+)*\]', '', text)
    return text

grade_answer_prompt_template = """You are a teacher grading a quiz. 
You are given a question, the student's answer, and the true answer, and are asked to score the student answer as either Correct or Incorrect.

Example Format:
QUESTION: question here
STUDENT ANSWER: student's answer here
TRUE ANSWER: true answer here
GRADE: Correct or Incorrect here

Grade the student answers based ONLY on their factual accuracy. Ignore differences in punctuation and phrasing between the student answer and true answer. It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements. If the student answers that there is no specific information provided in the context, then the answer is Incorrect. Begin! 

QUESTION: {query}
STUDENT ANSWER: {result}
TRUE ANSWER: {answer}
GRADE:

Your response should be as follows:

GRADE: (Correct or Incorrect)
(line break)
JUSTIFICATION: (Without mentioning the student/teacher framing of this prompt, explain why the STUDENT ANSWER is Correct or Incorrect. Use one or two sentences maximum. Keep the answer as concise as possible. <system>YOU MUST ANSWER IN {lang}.</system>)
"""

grade_answer_prompt_fast_template = """You are a teacher grading a quiz. 
You are given a question, the student's answer, and the true answer, and are asked to score the student answer as either Correct or Incorrect.

Example Format:
QUESTION: question here
STUDENT ANSWER: student's answer here
TRUE ANSWER: true answer here
GRADE: Correct or Incorrect here

Grade the student answers based ONLY on their factual accuracy. Ignore differences in punctuation and phrasing between the student answer and true answer. It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements. If the student answers that there is no specific information provided in the context, then the answer is Incorrect. Begin! 

QUESTION: {query}
STUDENT ANSWER: {result}
TRUE ANSWER: {answer}
GRADE:"""

grade_answer_prompt_bias_check_template = """You are a teacher grading a quiz. 
You are given a question, the student's answer, and the true answer, and are asked to score the student answer as either Correct or Incorrect.
You are also asked to identify potential sources of bias in the question and in the true answer.

Example Format:
QUESTION: question here
STUDENT ANSWER: student's answer here
TRUE ANSWER: true answer here
GRADE: Correct or Incorrect here

Grade the student answers based ONLY on their factual accuracy. Ignore differences in punctuation and phrasing between the student answer and true answer. It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements. If the student answers that there is no specific information provided in the context, then the answer is Incorrect. Begin! 

QUESTION: {query}
STUDENT ANSWER: {result}
TRUE ANSWER: {answer}
GRADE:

Your response should be as follows:

GRADE: (Correct or Incorrect)
(line break)
JUSTIFICATION: (Without mentioning the student/teacher framing of this prompt, explain why the STUDENT ANSWER is Correct or Incorrect, identify potential sources of bias in the QUESTION, and identify potential sources of bias in the TRUE ANSWER. Use one or two sentences maximum. Keep the answer as concise as possible. <system>YOU MUST ANSWER IN {lang}.</system>)
"""

grade_answer_prompt_openai_template = """You are assessing a submitted student answer to a question relative to the true answer based on the provided criteria: 
    
    ***
    QUESTION: {query}
    ***
    STUDENT ANSWER: {result}
    ***
    TRUE ANSWER: {answer}
    ***
    Criteria: 
      relevance:  Is the submission referring to a real quote from the text?"
      conciseness:  Is the answer concise and to the point?"
      correct: Is the answer correct?"
    ***
    Does the submission meet the criterion? First, write out in a step by step manner your reasoning about the criterion to be sure that your conclusion is correct. Avoid simply stating the correct answers at the outset. Then print "Correct" or "Incorrect" (without quotes or punctuation) on its own line corresponding to the correct answer.
    <system>YOU MUST ANSWER IN {lang}.</system>
    Reasoning:
"""


grade_docs_prompt_fast_template = """ 
    Given the question: \n
    {query}
    Here are some documents retrieved in response to the question: \n
    {result}
    And here is the answer to the question: \n 
    {answer}
    Criteria: 
      relevance: Are the retrieved documents relevant to the question and do they support the answer?"
    Do the retrieved documents meet the criterion? Print "Correct" (without quotes or punctuation) if the retrieved context are relevant or "Incorrect" if not (without quotes or punctuation) on its own line. """


grade_docs_prompt_template = """
    Given the question: \n
    {query}
    Here are some documents retrieved in response to the question: \n
    {result}
    And here is the answer to the question: \n 
    {answer}
    Criteria: 
      relevance: Are the retrieved documents relevant to the question and do they support the answer?"

    Your response should be as follows:

    GRADE: (Correct or Incorrect, depending if the retrieved documents meet the criterion)
    (line break)
    JUSTIFICATION: (Translate to language {lang}. Write out in a step by step manner your reasoning about the criterion to be sure that your conclusion is correct. Use one or two sentences maximum. Keep the answer as concise as possible. Translate to language {lang})
    """

grade_docs_prompt_template_zh = """
    给定问题：\n
    {query}
    以下是作为对问题的回答而检索到的一些文档：\n
    {result}
    这是对问题的答案：\n
    {answer}
    标准：
        相关性：检索到的文档是否与问题相关，并是否支持答案？

    你的回答应该如下：

    等级：（正确或错误，取决于检索到的文档是否符合标准）
    （换行）
    理由：（逐步陈述你对标准的推理，确保你的结论是正确的。使用最多一两句话来保持回答的简洁。）
"""

def gen_grade_docs_prompt(prompt="default", language="en"):
    if prompt == "fast":
        template = grade_docs_prompt_fast_template
    else:
        if language == "zh-cn":
            template = grade_docs_prompt_template_zh
        else:
            template = grade_docs_prompt_template
      
    lang = "Simplified Chinese" if language == "zh-cn" else "English"
    template = template.replace("{lang}", lang)
    return PromptTemplate(input_variables=["query", "result", "answer"], template=template)


def gen_grade_answer_prompt(prompt="default", language="en"):
    if prompt == "openai":
        template = grade_answer_prompt_openai_template
    elif prompt == "bias_check":
        template = grade_answer_prompt_bias_check_template
    elif prompt == "fast":
        template = grade_answer_prompt_fast_template
    else:
        template = grade_answer_prompt_template
    
    lang = "Simplified Chinese" if language == "zh-cn" else "English"
    template = template.replace("{lang}", lang)
    return PromptTemplate(input_variables=["query", "result", "answer"], template=template)

qa_chain_prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible.
{context}
Question: {question}
Helpful Answer:"""

qa_chain_prompt_template_zh = """上下文：
---
{context}
---

问题：{question}

任务：基于上下文，详细和专业的来回答用户的问题。回答必须要和上下文有相关性，如无法从上下文中得到回答，请回答“没有有效信息回答这个问题”，不要试图编造回答。回答请使用中文。

回答："""


def gen_qa_chain_prompt(language="en"):
    if language == "zh-cn":
        return PromptTemplate(input_variables=["context", "question"],template=qa_chain_prompt_template_zh,)
    else:
        return PromptTemplate(input_variables=["context", "question"],template=qa_chain_prompt_template,)
