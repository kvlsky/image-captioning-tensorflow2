import plotly.graph_objects as go
import pandas as pd

data = pd.read_csv(
    '/Users/andriikoval/Documents/Uni/WS19-20/ADL/Project/advanced-deep-learning-project/run-20200116-114934_train-tag-Bleu_2(1).csv')

x = data['Step']
y = data['Value']


# Create traces
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y,
                         mode='lines',
                         name='lines'))
# fig.add_trace(go.Scatter(x=random_x, y=random_y1,
#                     mode='lines+markers',
#                     name='lines+markers'))

fig.update_layout(title='Bleu-4',
                  xaxis_title='epochs',
                  yaxis_title='score')

fig.show()

data_l2 = pd.read_csv(
    '/Users/andriikoval/Documents/Uni/WS19-20/ADL/Project/advanced-deep-learning-project/l2_reg_loss.csv')
x_l2 = data_l2['Step']
y_l2 = data_l2['Value']

data_loss = pd.read_csv(
    '/Users/andriikoval/Documents/Uni/WS19-20/ADL/Project/advanced-deep-learning-project/loss.csv')
x_loss = data_loss['Step']
y_loss = data_loss['Value']

fig_l2 = go.Figure()
fig_l2.add_trace(go.Scatter(x=x_l2, y=y_l2,
                            mode='lines',
                            name='l2 loss'))
fig_l2.add_trace(go.Scatter(x=x_loss, y=y_loss,
                            mode='lines',
                            name='loss'))
# fig.add_trace(go.Scatter(x=random_x, y=random_y2,
#                     mode='markers', name='markers'))
fig_l2.update_layout(title='losses',
                     xaxis_title='epoch',
                     yaxis_title='loss',
                     font=dict(
                         family="Courier New, monospace",
                         size=32,
                         color="#7f7f7f"
                     ))

fig_l2.show()
# fig_l2.write_image("fig_l2.jpg")


data_lr = pd.read_csv(
    '/Users/andriikoval/Documents/Uni/WS19-20/ADL/Project/advanced-deep-learning-project/learning_rate.csv')
x_lr = data_lr['Step']
y_lr = data_lr['Value']

fig_lr = go.Figure()
fig_lr.add_trace(go.Scatter(x=x_lr, y=y_lr,
                            mode='lines',
                            name='lr'))
fig_lr.update_layout(title='lr',
                     xaxis_title='epoch',
                     yaxis_title='lr rate',
                     font=dict(
                         family="Courier New, monospace",
                         size=32,
                         color="#7f7f7f"
                     ))

fig_lr.show()
fig_lr.write_image("fig_lr.jpg")
