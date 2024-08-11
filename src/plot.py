import matplotlib
import matplotlib.pyplot as plt

from src.utils import History, filename_datetime


def plot_history(
    history: History,
    model_name: str,
    save_location: str = "output",
    title: str | None = None,
):
    """
    Plot the training history curves.

    Parameters
    ----------
    history: History
        The training history data.
    model_name: str
        Name of the model being plotted, used in plot title and saved filename.
    save_location: str = "output"
        Where to save the plot.
    title: str | None = None
        Whether to override the default title.
    """
    # configure matplotlib
    matplotlib.use("Agg")
    plt.style.use("ggplot")

    x_ticks = [x + 1 for x in range(len(history.train_accuracy))]

    fig = plt.figure(figsize=(12, 6))

    if title is not None:
        plt.suptitle(title)
    else:
        plt.suptitle(
            f"{model_name.capitalize()} Training ({filename_datetime()})", size=16
        )

    # plot loss on training and validation sets
    _ = fig.add_subplot(1, 2, 1)
    plt.title("Loss During Training", size=12)
    plt.plot(x_ticks, history.train_loss, label="training data")
    plt.plot(x_ticks, history.test_loss, label="validation data")
    plt.legend(facecolor="white", framealpha=1)
    plt.grid(True)
    plt.gca().xaxis.get_major_locator().set_params(integer=True)
    plt.ylabel("loss")
    plt.xlabel("epoch")

    # plot accuracy on training and validation sets
    ax = fig.add_subplot(1, 2, 2)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    plt.title("Accuracy During Training", size=12)
    plt.plot(x_ticks, history.train_accuracy, label="training data")
    plt.plot(x_ticks, history.test_accuracy, label="validation data")
    plt.legend(facecolor="white", framealpha=1)
    plt.grid(True)
    plt.gca().xaxis.get_major_locator().set_params(integer=True)
    plt.ylabel("accuracy")
    plt.xlabel("epoch")

    plt.savefig(
        f"{save_location.rstrip('/')}/{model_name}_fig_{filename_datetime()}.png"
    )
