INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""

USER_PROMPT_TEMPLATE = """
QUESTION:
{question}

CONTEXT:
{context}
""".strip()


class RAGBase:

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=USER_PROMPT_TEMPLATE,
        course='llm-zoomcamp',
        model='gpt-5.4-mini'
    ):
        self.index = index
        self.client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.course = course
        self.model = model

    def search(self, question):
        boost_dict = {'question': 2.0, 'section': 0.5}
        filter_dict = {'course': self.course}

        return self.index.search(
            question,
            boost_dict=boost_dict,
            filter_dict=filter_dict,
            num_results=5
        )

    def build_context(self, search_result):
        lines = []
        
        for doc in search_result:
            lines.append(doc['section'])
            lines.append('Q: ' + doc['question'])
            lines.append('A: ' + doc['answer'])
            lines.append('')
        
        return '\n'.join(lines).strip()

    def build_prompt(self, question, search_results):
        context = self.build_context(search_results)
        prompt = self.prompt_template.format(question=question, context=context)

        return prompt.strip()

    def ask_llm(self, instructions: str, user_prompt: str, model: str='gpt-5.4-mini'):

        message_history = [
            {'role': 'developer', 'content': instructions},
            {'role': 'user', 'content': user_prompt},
        ]

        response = self.client.responses.create(
            model=model,
            input=message_history,
        )

        return response.output_text

    def rag(self, question:str):
        search_results = self.search(question)
        prompt = self.build_prompt(question, search_results)

        return self.ask_llm(self.instructions, prompt, self.model)
