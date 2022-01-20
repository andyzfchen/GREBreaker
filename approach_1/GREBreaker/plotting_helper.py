import numpy as np
import matplotlib.pyplot as plt

def plot_wp_accuracies(data_name, accuracy_types, d_dim, filename, title="Test"):
    n_epoch = -1
    #line_colors = [ "r", "g", "b", "m" ]
    line_styles = [ "-", "--" ] 
    accuracy_names = [ "Train", "Validation" ] 
    fig, ax = plt.subplots(figsize=(4, 3))

    for ii, accuracy_type in enumerate(accuracy_types):
      acc_vec = np.load("../cache/word_prediction/"+data_name+"_glove"+str(d_dim)+"_"+accuracy_type+"_acc.npy")
      if n_epoch == -1:
        n_epoch = len(acc_vec)
        
      position = np.arange(n_epoch)
      ax.plot(position, acc_vec, label=accuracy_names[ii])

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_xlim(position[0], position[-1])
    ax.set_ylabel("Accuracy")
    #ax.set_yscale('log')
    ax.set_ylim(0,1)
    ax.legend(loc="lower right")
    plt.savefig("../figures/"+filename+".pdf", bbox_inches="tight", pad_inches=0.2)
    plt.close()
