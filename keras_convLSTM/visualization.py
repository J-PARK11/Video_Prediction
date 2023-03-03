import io
import imageio
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image, display
from ipywidgets import widgets, Layout, HBox, VBox

def multi_frame_visualize(dataset, path='',cmap_param='gray'):
    # Construct a figure on which we will visualize the images.
    fig, axes = plt.subplots(4, 5, figsize=(10, 8))

    # Plot each of the sequential images for one data example.
    for idx, ax in enumerate(axes.flat):
        ax.imshow(np.squeeze(dataset[idx]), cmap=cmap_param)
        ax.set_title(f"Frame {idx + 1}")
        ax.axis("off")

    if path != '':
        plt.savefig(path)
    plt.show()

def prediction_visualize(true, pred, path='', cmap_param='gray'):
    # Construct a figure for the original and new frames.
    fig, axes = plt.subplots(2, 10, figsize=(20, 4))

    # Plot the original frames.
    for idx, ax in enumerate(axes[0]):
        ax.imshow(np.squeeze(true[idx]), cmap=cmap_param)
        ax.set_title(f"Frame {idx + 11}")
        ax.axis("off")

    # Plot the new frames.
    for idx, ax in enumerate(axes[1]):
        np.squeeze(pred[idx])*255
        ax.imshow(np.squeeze(pred[idx]), cmap=cmap_param)
        ax.set_title(f"Frame {idx + 11}")
        ax.axis("off")

    # Display the figure.
    if path != '':
        plt.savefig(path)
    plt.show()



# Display Eval Videos
# ---------------------------------------------------------------- #
def create_eval_video(true_frame, pred_frame, n):
    true_videos = create_multi_video(true_frame, n)
    pred_videos = create_multi_video(pred_frame, n)

    return [true_videos, pred_videos]

def display_evals_video(video):
    display_multi_video(video[0],'true')
    display_multi_video(video[1],'pred')
# ---------------------------------------------------------------- #

# Display Single Videos
# ---------------------------------------------------------------- #
def create_single_video(dataset):
    videos = []
    frames = dataset
    current = np.squeeze(frames)
    current = current[..., np.newaxis] * np.ones(3)
    current = (current * 255).astype(np.uint8)
    current = list(current)

    with io.BytesIO() as gif:
        imageio.mimsave(gif, current, "GIF", fps=5)
        videos.append(gif.getvalue())
    return videos

def display_single_video(video, name='single_video'):
    print(f'{name}')
    box = HBox([widgets.Image(value=video[0])])
    display(box)
# ---------------------------------------------------------------- #

# Display Multi Videos
# ---------------------------------------------------------------- #
def create_multi_video(dataset, n):
    videos = []
    for frames in dataset[:n]:
        current = np.squeeze(frames)
        current = current[..., np.newaxis] * np.ones(3)
        current = (current * 255).astype(np.uint8)
        current = list(current)

        with io.BytesIO() as gif:
            imageio.mimsave(gif, current, "GIF", fps=5)
            videos.append(gif.getvalue())
    return videos

def display_multi_video(video, name='multi_videos'):
    print(f'{name}')
    widget = [widgets.Image(value=video[i]) for i in range(len(video))]
    box = HBox(widget)
    display(box)
# ---------------------------------------------------------------- #