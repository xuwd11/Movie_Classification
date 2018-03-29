import time

class CountRequest:
    def __init__(self, limit=40, stop=10):
        self.limit = limit
        self.stop = stop
        self.count = 0
        
    def add(self):
        self.count += 1
        if self.count % self.limit == 0:
            print('Sleep {} seconds...'.format(self.stop))
            time.sleep(self.stop)
        return self