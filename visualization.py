import io
import imageio
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image, display
from ipywidgets import widgets, Layout, HBox


def multi_frame_visualize(dataset, cmap_param='gray'):
    # Construct a figure on which we will visualize the images.
    fig, axes = plt.subplots(4, 5, figsize=(10, 8))

    # Plot each of the sequential images for one random data example.
    for idx, ax in enumerate(axes.flat):
        ax.imshow(np.squeeze(dataset[idx]), cmap=cmap_param)
        ax.set_title(f"Frame {idx + 1}")
        ax.axis("off")
    plt.show()


def prediction_visualize(true, pred, cmap_param='gray'):
    # Construct a figure for the original and new frames.
    fig, axes = plt.subplots(2, 10, figsize=(20, 4))

    # Plot the original frames.
    for idx, ax in enumerate(axes[0]):
        ax.imshow(np.squeeze(true[idx]), cmap=cmap_param)
        ax.set_title(f"Frame {idx + 11}")
        ax.axis("off")

    # Plot the new frames.
    new_frames = pred[10:, ...]
    for idx, ax in enumerate(axes[1]):
        # np.squeeze(pred[idx])*255
        ax.imshow(np.squeeze(pred[idx]), cmap=cmap_param)
        ax.set_title(f"Frame {idx + 11}")
        ax.axis("off")

    # Display the figure.
    plt.show()

def create_videos(dataset):
    # Select a few random examples from the dataset.
    examples = dataset[np.random.choice(range(len(dataset)), size=5)]

    # Iterate over the examples and predict the frames.
    predicted_videos = []
    for example in examples:
        # Pick the first/last ten frames from the example.
        frames = example[:10, ...]
        original_frames = example[10:, ...]
        new_predictions = np.zeros(shape=(10, *frames[0].shape))

        # Create and save GIFs for each of the ground truth/prediction images.
        for frame_set in [original_frames, new_predictions]:
            # Construct a GIF from the selected video frames.
            current_frames = np.squeeze(frame_set)
            current_frames = current_frames[..., np.newaxis] * np.ones(3)
            current_frames = (current_frames).astype(np.uint8)
            current_frames = list(current_frames)

            # Construct a GIF from the frames.
            with io.BytesIO() as gif:
                imageio.mimsave(gif, current_frames, "GIF", fps=5)
                predicted_videos.append(gif.getvalue())
    return predicted_videos

def display_video(video):
    print(" Truth\tPrediction")
    for i in range(0, len(video), 2):
        # Construct and display an `HBox` with the ground truth and prediction.
        box = HBox(
            [
                widgets.Image(value=video[i]),
                widgets.Image(value=video[i + 1]),
            ]
        )
        display(box)