import matplotlib.pyplot as plt

def overall(project, name, format = "png", save = True, plot = True):
    """
        plots the first 10 traces of the project
    """
    plt.figure(figsize=(7,5))
    for i in range(100):
        plt.plot(project.waves[i])
    plt.title(f"First 100 traces of {name}")
    plt.xlabel("Sample Number")
    plt.ylabel("Voltage")
    if save:
        plt.savefig(name + "_overall." + format, format=format)
    if plot:
        plt.show()
    
def ddla_results(correct_key, attack_result, trace_num, key_range, name, bnum, format = "png", save = True, plot = True):
    """
        plots the results of a ddla attack
    """
    fig, axs = plt.subplots(1, 3, figsize=(20,5))
    fig.suptitle(f"Attack Result for {trace_num} traces on subkey {bnum}") # type: ignore
    axs[0].set_title("Accuracy") # type: ignore
    axs[2].set_title("Loss") # type: ignore
    axs[1].set_title("Sensitivity") # type: ignore

    num = 0
    best_guess_key = attack_result.best_guesses()[bnum]['guess']
    for key in key_range: # type: ignore
        if num == 0 and key != correct_key and key != best_guess_key:
            axs[0].plot(attack_result.accuracy[bnum][key], color = 'gray', label = 'wrong key', linewidth = 0.5) # type: ignore
            axs[1].plot(attack_result.metrics[bnum][key], color = 'gray', label = 'wrong key', linewidth = 0.5) # type: ignore
            axs[2].plot(attack_result.loss[bnum][key], color = 'gray', label = 'wrong key', linewidth = 0.5) # type: ignore
            num += 1
        else: 
            axs[0].plot(attack_result.accuracy[bnum][key], color = 'gray', linewidth = 0.5) # type: ignore
            axs[1].plot(attack_result.metrics[bnum][key], color = 'gray', linewidth = 0.5) # type: ignore
            axs[2].plot(attack_result.loss[bnum][key], color = 'gray', linewidth = 0.5) # type: ignore

    axs[0].plot(attack_result.accuracy[bnum][correct_key], color = 'red', label = 'correct key') # type: ignore
    axs[1].plot(attack_result.metrics[bnum][correct_key], color = 'red', label = 'correct key') # type: ignore
    axs[2].plot(attack_result.loss[bnum][correct_key], color = 'red', label = 'correct key') # type: ignore

    if best_guess_key != correct_key:    
        axs[0].plot(attack_result.accuracy[bnum][best_guess_key], color = 'blue', label = 'best guess key') # type: ignore
        axs[1].plot(attack_result.metrics[bnum][best_guess_key], color = 'blue', label = 'best guess key') # type: ignore
        axs[2].plot(attack_result.loss[bnum][best_guess_key], color = 'blue', label = 'best guess key') # type: ignore
        
    axs[0].legend() # type: ignore
    axs[1].legend() # type: ignore
    axs[2].legend() # type: ignore
    axs[0].set_xlabel("Epoch") # type: ignore
    axs[1].set_xlabel("Time Samples") # type: ignore
    axs[2].set_xlabel("Epoch") # type: ignore
    if save:
        plt.savefig(f'{name}_key{bnum}_ddla_result.{format}', format=format)
    if plot:
        plt.show()
