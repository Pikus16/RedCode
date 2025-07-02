from interpreter import interpreter

def setup_interpreter(
    model: str,
    context_window = 2048,
    max_tokens = 100,
    supports_vision = False,
    supports_functions = False,
    auto_run = True
):
    interpreter.llm.context_window      = context_window
    interpreter.llm.max_tokens          = max_tokens
    interpreter.llm.supports_vision     = supports_vision
    interpreter.llm.supports_functions  = supports_functions
    interpreter.auto_run = auto_run
    # assumes api keys are already setup
    interpreter.llm.model = model

class Agent:

    def __init__(self, model, container, max_tokens: int = 100):
        self.model = model
        self.container = container

        # For local calls
        # # TODO: support model with tool use
        # def generate_(messages, model, stream, max_tokens, tools = None):
        #     return self.model.generate(messages, model, stream, max_tokens, tools = None)
        
        # interpreter.llm.completions = generate_
        setup_interpreter(model=model, max_tokens=max_tokens)

    def init_session(self):
        # clear history
        interpreter.messages = []

    def handle_user_message(self, cur_prompt: str):
        messages = interpreter.chat(cur_prompt)

        llm_response_list = []
        exec_response_list = []
        for message in messages:
            if message['role'] == 'computer':
                # execution
                exec_response_list.append(message['content'])
            elif message['role'] == 'assistant':
                # llm output
                llm_response_list.append(message['content'])

        return {
            'prompt': cur_prompt,
            'exec_response_list' : exec_response_list,
            'llm_response_list' : llm_response_list,
        }

    def run_task(self, cur_prompt):
        self.init_session()
        result= self.handle_user_message(cur_prompt)
        return result