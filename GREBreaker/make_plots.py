from plotting_helper import plot_wp_accuracies
import os

data_names = [ "SAT", "SCS", "501sc" ]
accuracy_types = [ "train", "val" ]
d_dims = [ 50, 100, 200, 300 ]

if not os.path.exists("../figures"):
  os.mkdir("../figures")

for data_name in data_names:
  for d_dim in d_dims:
    plot_wp_accuracies(data_name, accuracy_types, d_dim, data_name+"_d"+str(d_dim), title=data_name+" Accuracies ($d = "+str(d_dim)+"$)")
