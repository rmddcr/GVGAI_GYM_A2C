#results_file = open("all_games.txt", "w")
from os import listdir
from os.path import isfile, join
onlyfiles = ["gvgai-"+f[0:-3] for f in listdir("./gym_gvgai/envs/games/")]
print(onlyfiles)