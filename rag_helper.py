import logging


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

logger = logging.getLogger(__name__)


class RAGBase:

    def __init__(
        self,
        index,
        llm_client,
        context_builder,
        instructions=INSTRUCTIONS,
        prompt_template=USER_PROMPT_TEMPLATE,
        course='llm-zoomcamp',
        model='gpt-5.4-mini'
    ):
        self.index = index
        self.client = llm_client
        self.instructions = instructions
        self.context_builder = context_builder
        self.prompt_template = prompt_template
        self.course = course
        self.model = model

    def search(self, question, boost_dict, filter_dict, num_results=5):
        return self.index.search(
            question,
            boost_dict=boost_dict,
            filter_dict=filter_dict,
            num_results=num_results
        )

    def build_prompt(self, question, search_results):
        context = self.context_builder.build(search_results)
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

        logger.debug('Number of input tokens consumed: %d', response.usage.input_tokens)

        return response.output_text, response.usage.input_tokens

    def rag(self, question:str, boost_dict={}, filter_dict={}):
        search_results = self.search(question, boost_dict, filter_dict)

        logging.debug('Search results: %s', search_results)

        prompt = self.build_prompt(question, search_results)

        return self.ask_llm(self.instructions, prompt, self.model)
