from multiprocessing import Queue
from threading import Thread
import time


def data_generator():
    for ex in range(380):
        yield ex


class My(object):
    def __init__(self):
        self.bucketing_cache_size = 5
        self.batch_size = 50

        self.example_queue = Queue(5*self.batch_size)
        self.batch_queue = Queue(5)
    
    def setup(self):
        self.finished_reading = False
        self.prepare_to_end = False
        self.ready_to_end = False
        self.stop = False

        self.example_q_thread = Thread(target=self.fill_example_queue)
        self.example_q_thread.daemon = True
        self.example_q_thread.start()
        
        self.batch_q_thread = Thread(target=self.fill_batch_queue)
        self.batch_q_thread.daemon = True
        self.batch_q_thread.start()


    def fill_example_queue(self):
        input_gen = data_generator()
        while True:
            try:
                example = next(input_gen)
            except StopIteration:
                self.finished_reading = True
                print('data file reading finish, example thread will dead.')
                break
            
            self.example_queue.put(example)


    def fill_batch_queue(self):
        while True:
            if self.prepare_to_end:
                self.ready_to_end = True
                break
            inputs = []
            for _ in range(self.batch_size * self.bucketing_cache_size):
                if self.example_queue.qsize() == 0 and self.finished_reading:
                    self.prepare_to_end = True
                    break
                ex = self.example_queue.get()
                inputs.append(ex)
            
            batches = []
            for i in range(0, len(inputs), self.batch_size):
                end_i = min(i+self.batch_size, len(inputs))
                batches.append(inputs[i:end_i])
            for b in batches:
                self.batch_queue.put(b)


    def next_batch(self):
        if self.ready_to_end and self.batch_queue.qsize() == 0:
            self.stop = True
            raise StopIteration('no more batch')
        if self.batch_queue.qsize() == 0:
            print('current batch queue size: %i, example queue size: %i' % (self.batch_queue.qsize(), self.example_queue.qsize()))
        data = self.batch_queue.get()
        return data


def one_epoch(hehe):
    # read all data only once
    hehe.setup()
    total = 0
    while True:
        try:
            data = hehe.next_batch()
            print(data)
            total += sum(data)
        except StopIteration:
            assert total == 72010, 'error: total=%d' % total
            break


def multi_epoch():
    # read all data for num_epoch times
    hehe = My()
    num_epoch = 1000
    for _ in range(num_epoch):
        print('begin new epoch')
        print('*'*50)
        one_epoch(hehe)

if __name__ == '__main__':
    multi_epoch()
    #one_epoch()

