## This is the list of core prompts that I have used to build this app.

## Chat GPT - Codex - GPT 5.4/5.5

## First prompt in Plan mode.
I have just installed Ollama on my local machine.
Create a plan to build a CLI interface app using python that will allow for me to run an LLM model locally like "Llama 3".
It should allow a user to ask questions and provide them with answers.
Additionally, i want a way to ensure that the LLM model will only use the paper "C:\Users\User\Documents\ATUCourse\BigDataArchitecture\Assignment3\CLI Papaer -Research Landscape of Agentic AI and LLM - Apps, Challenges and Future Direction.pdf" as its source for answering questions. Tell me how i can approach this.


## Second prompt in Plan mode, after creation of app and some tweaks.

Currently the app.py file contains the entire code. This is difficult to follow.
Break down the file into easier to digest parts that make logical sense, follow best practice for Python development.
All functionality should remain the same for the user.

## Third prompt, more debugging

In the ollama_client.py file, i have added a console.print statement, I want this to only print when user has passed the --debug flag similar to the checks in the commands.py file. Implement this in a clean, concise manner.

## Fourth prompt, system prompt update request

Below is my current system prompt. It's goal is to only return an answer to the user if there is relevant information in the documents. If not it should return a standard "cannot answer" sentance. Currently there is an issue where the LLM is returning partial answers followed by the "cannot answer" sentance. This is incorrect. Analyse the prompt, its meaning, the goal and provide me with an improved prompt for me to use as the core System prompt. Remember that the overall goal of this application is to provide the user with answers to their questions ONLY if they can be found within the provided document.

system_prompt = (
        "You are a careful research assistant. Answer the user's question using only the "
        "paper excerpts provided in the context. "
        "When you find an answer in the paper include page citations in the "
        "form (p. 3) or (pp. 3, 5) for every factual answer. Do not use outside knowledge. If the "
        "context does not contain enough evidence, answer exactly: "
        "'I cannot answer that from the provided paper.' with NO page citations."
        
    )