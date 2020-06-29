import sys
import string
import random

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def write():
    print('Creating new text file') 

    name = (id_generator())+'.txt'  # Name of text file coerced with +.txt

    try:
        file = open(name,'w')   # Trying to create a new file or open one
        file.close()

    except:
        print('Something went wrong! Can\'t tell what?')
        sys.exit(0) # quit Python

write()
