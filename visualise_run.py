import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def add_running_ave(data, column, window_width=10):
    cumsum_vec = np.cumsum(np.insert(data[column], 0, 0))
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    ma_vec = np.append(np.zeros(int(window_width/2)), ma_vec)
    ma_vec = np.append(ma_vec, np.zeros(int(window_width/2)))
    running_ave_df = pd.DataFrame(ma_vec, columns=[f"{column}_running_ave"])

    return pd.concat([data, running_ave_df], axis=1)

def visualise_latest():
    folder_name=sorted(os.listdir("Outputs"))[-1]

    sns.set_theme(style="darkgrid")

    frame_data = pd.read_csv(f"Outputs/{folder_name}/frame_data.csv")
    cropped_frame_data = frame_data.loc[frame_data["loss"] > 0.0]
    # cropped_frame_data = cropped_frame_data.loc[frame_data["loss"] < 13]
    cropped_frame_data = cropped_frame_data.reset_index(drop=True)
    episode_data = pd.read_csv(f"Outputs/{folder_name}/episode_data.csv")

    # Reward moving average
    episode_data = add_running_ave(episode_data, "total_reward", window_width=10)
    cropped_frame_data = add_running_ave(cropped_frame_data, "loss", window_width=round(len(cropped_frame_data)/10))
    
    # Reward plot
    sns.scatterplot(episode_data, x = np.arange(0, len(episode_data["total_reward"]-1)), y="total_reward", hue="episode_length", marker="o")
    sns.lineplot(episode_data, x = np.arange(0, len(episode_data["total_reward"]-1)), y="total_reward_running_ave")
    plt.savefig(f"Outputs/{folder_name}/reward_over_time.svg")
    plt.show()

    # Loss plot
    sns.scatterplot(cropped_frame_data, x="frame_number", y="loss", marker="o", legend=False, alpha=0.3, hue="reward")
    sns.lineplot(cropped_frame_data, x="frame_number", y="loss_running_ave", alpha=0.7)
    # axes.set_ylim(-1, 13)
    plt.savefig(f"Outputs/{folder_name}/loss_over_time.svg")
    plt.show()

if __name__ == "__main__":
    visualise_latest()