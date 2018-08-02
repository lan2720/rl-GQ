"""
A simple python wrapper for the standard mult-bleu.perl 
script used in machine translation/captioning models.

References
     NeuralTalk (https://github.com/karpathy/neuraltalk)

"""

reference = '1 2 3' #'Two person be in a small race car drive by a green hill .'
output = '1 2 3 4' #'Two person in race uniform in a street car .'

with open('output', 'w') as output_file:
    output_file.write(reference)

with open('reference', 'w') as reference_file:
    reference_file.write(output)

from os import system
system('./multi-bleu.perl reference < output')
