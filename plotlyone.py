import plotly.graph_objects as go

import plotly.express as px

import numpy as np
import pandas as pd
import random
test_set = np.load('./test/test.npy')
test_label = np.load('./test/test_label.npy')

time_samples = [i for i in range(0, 150, 6)] + [i for i in range(150, 240, 5)] + \
               [i for i in range(240, 290, 2)] + [i for i in range(290, 300)]
features = [i for i in range(10)]
test_set = test_set[:, time_samples, :][:, :, features]

columns = ['L_x', 'L_y', 'L_z', 'C_x', 'C_y', 'C_z', 'F_x', 'F_y', 'F_z', 'confidence',
           'gaze_norm_pos_x', 'gaze_norm_pos_y', 'norm_pos_x', 'norm_pos_y', 'diameter',
           'ellipse_center_x', 'ellipse_center_y', 'ellipse_axis_a', 'ellipse_axis_b',
           'ellipse_angle', 'diameter_3d', 'model_confidence', 'model_id', 'sphere_center_x',
           'sphere_center_y', 'sphere_center_z', 'sphere_radius', 'circle_3d_center_x',
           'circle_3d_center_y', 'circle_3d_center_z', 'circle_3d_normal_x', 'circle_3d_normal_y',
           'circle_3d_normal_z', 'circle_3d_radius', 'theta', 'phi', 'projected_sphere_center_x',
           'projected_sphere_center_y', 'projected_sphere_axis_a', 'projected_sphere_axis_b']
# df = pd.DataFrame(columns=['index', 'frame'] + columns[:10] + ['label'])

# for i in range(200,400):
#     instance = test_set[i, :, :]
#     label = test_label[i]
#     for t in range(instance.shape[0]):
#         df.loc[(i, t), :] = instance[t, :].tolist() + [label]


# def draw_random():
#     df.set_index(['index', 'frame'], inplace=True)
#     i = random(test_set.shape[0])
#     label = test_label[i]
#     for t in range(test_set.shape[1]):
#         df.loc[(0, t), :] = test_set[i, t, :].tolist() + [label]
#     df.reset_index(inplace=True)


i = random.randrange(test_set.shape[0])
instance = test_set[i, :, :]
label = test_label[i]

x_lines = list()
y_lines = list()
z_lines = list()

for p in range(test_set.shape[1]):
    x_lines.append(instance[p, 0])
    y_lines.append(instance[p, 2])
    z_lines.append(instance[p, 1])

    x_lines.append(instance[p, 0+3])
    y_lines.append(instance[p, 2+3])
    z_lines.append(instance[p, 1+3])

    x_lines.append(None)
    y_lines.append(None)
    z_lines.append(None)


camera_plot = go.Scatter3d(
    x=instance[:, 3], y=instance[:, 5], z=instance[:, 4],
    line=dict(color='darkblue', width=2),
    name='Camera Positions'
)

pog_line_plot = go.Scatter3d(
    x=x_lines, y=y_lines, z=z_lines,
    mode='lines',
    name='PoG Lines'
)

frames = []
for t in range(test_set.shape[1]):
    frames.append(go.Frame(data=[
        go.Scatter3d(
            x=instance[:t, 3], y=instance[:t, 5], z=instance[:t, 4],
            line=dict(color='darkblue', width=1),
            name='Camera Positions'),

        go.Scatter3d(
            x=x_lines[t*2:t*3], y=y_lines[t*2:t*3], z=z_lines[t*2:t*3],
            mode='lines',
            name='PoG Lines')
    ]))

fig = go.Figure(data=[
    go.Scatter3d(
        x=instance[:, 3], y=instance[:, 5], z=instance[:, 4],
        line=dict(color='darkblue', width=1),
        name='Camera Positions'),

    go.Scatter3d(
        x=x_lines, y=y_lines, z=z_lines,
        mode='lines',
        name='PoG Lines')],

    frames=frames,

    # layout_xaxis_range=[-25,-8],
    # layout_yaxis_range=[-11,1],
    # layout_zaxis_range=[0,4],

    layout=go.Layout(
        updatemenus=[dict(type="buttons",
                          buttons=[dict(label="Play", method="animate", args=[None])])])
)

fig.update_layout(scene_aspectmode='cube')
fig.update_layout(xaxis_range=[-30, 30])
fig.update_layout(yaxis_range=[-30, 30])
fig.show()
