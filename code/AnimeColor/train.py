from options import ModelOptions
from main import main

options = ModelOptions().parse()
options.mode = 0
main(options)
