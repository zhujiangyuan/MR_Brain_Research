class ProgramException(BaseException):
    def __init__(self,model,msg):
        self.model = model
        self.msg=msg
    def __str__(self):
        message = self.model + ':'
        message += self.msg
        return message
