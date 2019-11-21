from options import ModelOptions
from main import main

options = ModelOptions().parse()
options.mode = 2
main(options)
