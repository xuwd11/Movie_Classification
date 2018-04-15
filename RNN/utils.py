import sys

class ProgressBar:
    def __init__(self, total, display=True):
        self.cur = 0
        self.total = total
        self.prev = 0
        self.display = display
    
    def print(self, txt, end=None):
        if not self.display:
            return
        if end is None:
            print(txt)
        else:
            print(txt, end=end)
    
    def tick(self):        
        self.cur += 1
        cur_p = self.cur / self.total
        self.print(".", end="")
        if cur_p * 100 >= self.prev + 5:
            self.prev += 5
            self.print(self.prev, end="")
            if self.prev >= 100:
                self.print(" F")
        sys.stdout.flush()
    
    def reset(self):
        self.cur = 0
        self.prev = 0
    