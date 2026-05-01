## List of some questions/issues hit along the way of building out this project and then steps tried to rectify them.

- When testing and there is a refusal to answer, page citations still printed.
    - Looks to be print issue in chat() as the llm responds but it is a refusal message and the print happens regardless.
- For some questions like "Tell me about LLMs", the response is valid but it included the "I cannot answer that from the provided paper." at the end which is wrong.
    - Swapping the model from llama3 to qwen3.5:4b resolved for one question but not a follow up "Is there any critical infrastructure protection required with LLMs.". This highlights that issue is with the System Prompt I am using or further up with the resources.
    - Issue is around the System prompt and how it was originally structured it, so the LLM was providing basically a partial answer and then stating it did not have the relevant information to answer it. Essentially it was competing between two instructions from the System prompt. Using ChatGPT to improve my system prompt to try resolve the issue. So now will check if enough infomration first, only when that is valid will it continue to generating an answer.
- Documents are still being provided for random questions like "Who won the world cup most recently?" and being passed to the LLM to answer, where it correcty refuses.
    - Issue is that my max-distance check for the embeddings check for question to document was too high. So reducing from 0.8 (which was my default) to 0.5 seems to have resolved this.
- Wanted to test out different embedding models for the .pdf file. The two others i chose pulled fine but would fail during index.
    - Issue was actually related to chunk-word size. The default size i had worked fine for the origin nomic embedding model but it was too large for the mxbai and snowflake models. So solution was to just pass a smaller chunk word and overlap size.
    - Going to rerun the nomic with the same size chunks to allow for better comparison between the 3.