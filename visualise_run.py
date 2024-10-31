import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def add_running_ave(data, column, window_width=10):
    cumsum_vec = np.cumsum(np.insert(data[column], 0, 0))
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    # ma_vec = np.append(np.zeros(int(window_width/2)), ma_vec)
    # ma_vec = np.append(ma_vec, np.zeros(int(window_width/2)))
    running_ave_df = pd.DataFrame(ma_vec, columns=[f"{column}_running_ave"])

    return pd.concat([data, running_ave_df], axis=1)

def visualise_latest():
    folder_name=sorted(os.listdir("Outputs"))[-1]

    sns.set_theme(style="darkgrid")

    frame_data = pd.read_csv(f"Outputs/{folder_name}/frame_data.csv")
    episode_data = pd.read_csv(f"Outputs/{folder_name}/episode_data.csv")

    # Calculate moving averages
    episode_window_width = 10
    # reward_window_width = len(frame_data)/10
    episode_data = add_running_ave(episode_data, "total_reward", window_width=episode_window_width)
    # frame_data = add_running_ave(frame_data, "reward", window_width=reward_window_width)
    

    # Reward plot
    fig = plt.figure()
    fig.subplots_adjust(right=0.8)
    num_episodes = len(episode_data["total_reward"])
    ax = sns.scatterplot(episode_data, x = np.arange(0, num_episodes), y="total_reward", hue="episode_length", marker="o")
    fig.add_subplot(ax)
    fig.set_figwidth(9) 

    len_dif = episode_window_width
    start_point = round(len_dif/2)
    ax2 = sns.lineplot(episode_data, x = np.arange(start_point, len(episode_data["total_reward_running_ave"])+start_point), y="total_reward_running_ave")
    
    fig.add_subplot(ax2)
    plt.legend(title='Episode Length', bbox_to_anchor=(1.0, 0., 0.0, 1.1))
    plt.savefig(f"Outputs/{folder_name}/reward_over_time.svg")
    plt.show()

    # Length vs reward plot
    plt.clf()
    sns.scatterplot(episode_data, x ="episode_length", y="total_reward", hue="total_reward", marker="o", legend=False)
    plt.savefig(f"Outputs/{folder_name}/reward_vs_episode_length.svg")
    plt.show()
    
    # Loss plot
    sns.scatterplot(frame_data, x="frame_number", y="loss", marker="o", legend=False, alpha=0.75, hue="reward", linewidth=0.05)
    
    plt.savefig(f"Outputs/{folder_name}/loss_over_time.png")
    plt.show()

if __name__ == "__main__":
    visualise_latest()