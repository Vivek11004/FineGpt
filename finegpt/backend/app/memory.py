class Memory:
    def __init__(self):
        self.chat_history = {}

    def add(self, user_id: str, user_input: str, bot_reply: str):
        if user_id not in self.chat_history:
            self.chat_history[user_id] = []
        self.chat_history[user_id].append((user_input, bot_reply))

    def get(self, user_id: str):
        return self.chat_history.get(user_id, [])
