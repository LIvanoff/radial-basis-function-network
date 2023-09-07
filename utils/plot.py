import matplotlib.pyplot as plt


def plot_figure(pred, x, y):
    plt.clf()
    plt.scatter(x.numpy(), y.numpy(), marker='o', c='#4169E1', alpha=0.8, label='Groud truth')  # c='#7B68EE'
    plt.plot(x.numpy(), pred.detach().numpy(), 'r', label='Prediction')
    plt.grid(alpha=0.4)
    plt.draw()
    plt.gcf().canvas.flush_events()
