## List of some questions/issues hit along the way of building out this project and then steps tried to rectify them.

- When testing and there is a refusal to answer, page citations still printed.
    - Looks to be print issue in chat() as the llm responds but it is a refusal message and the print happens regardless.
- For some questions like "Tell me about LLMs", the response is valid but it included the "I cannot answer that from the provided paper." at the end which is wrong.
    - Swapping the model from llama3 to qwen3.5:4b resolved for one question but not a follow up "Is there any critical infrastructure protection required with LLMs.". This highlights that issue is with the System Prompt I am using or further up with the resources.
    - 