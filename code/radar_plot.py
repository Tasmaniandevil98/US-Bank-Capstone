# import numpy as np
# import matplotlib.pyplot as plt

# # Function to create a radar plot


# def radar_plot(categories, values, title):
#     num_vars = len(categories)

#     # Compute angle for each axis
#     angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

#     # The plot is circular, so we need to "complete the loop" and append the start point to the end.
#     values += values[:1]
#     angles += angles[:1]

#     # Plot
#     fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
#     ax.fill(angles, values, color='blue', alpha=0.25)
#     ax.plot(angles, values, color='blue', linewidth=2)

#     # Set the title and legend
#     ax.set_title(title, size=20, color='blue', y=1.1)
#     ax.set_yticklabels([])
#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(categories, size=12)

#     plt.show()


# # Example data
# categories = ['Emotional Int', 'Category 2',
#               'Category 3', 'Category 4', 'Category 5']
# # Example values (should be normalized between 0 and 1)
# values = [4, 3, 2, 5, 4]

# # Call the function to create the radar plot
# radar_plot(categories, values, 'Example Radar Plot')


import pygal
from pygal.style import Style

# Custom style to control colors
custom_style = Style(
    colors=('#4EF1F6', '#D25092', '#5A9797')
)
# custom_style = Style(
#     colors=('#D25092', '#5A9797', '#FFA500')
# )

# Example data
categories = ['Harmlessness', 'Emotional Intelligence',
              'Groundedness', 'Factuality']
# h, ei, g, f
values_gpt = [4.38, 3.9, 3.05, 5]
values_llama = [4.43, 3.99, 2, 4.71]
values_gemini = [4.27, 4.46, 2.65, 4.63]


# Create Radar plot
# radar_chart = pygal.Radar(fill=True, show_dots=True)
# radar_chart.title = 'LLM Evaluation Plot'
# radar_chart.x_labels = categories
# radar_chart.add('GPT-4', values_gpt, fill=True,
#                 style={'fill': 'green', 'stroke': 'green'})
# radar_chart.add('Llama2', values_llama, fill=True, style={
#                 'fill': 'black', 'stroke': 'black'})
# radar_chart.add('Gemini', values_gemini, fill=True, style={
#                 'fill': 'orange', 'stroke': 'orange'})
# radar_chart = pygal.Radar(fill=True, show_dots=True, style=custom_style)
radar_chart = pygal.Radar(fill=True, show_dots=True)
radar_chart.title = 'LLM Evaluation Plot'
radar_chart.x_labels = categories
radar_chart.add('GPT-4', values_gpt)
radar_chart.add('Llama2', values_llama)
radar_chart.add('Gemma', values_gemini)

# Save the plot to a file
radar_chart.render_to_file('radar_chart.svg')


# import matplotlib.pyplot as plt
# import numpy as np

# # Define the categories and values
# categories = ['Harmlessness', 'Emotional Intelligence',
#               'Groundedness', 'Factuality']
# values_gpt = [4.514, 3.48, 3.05, 4.5]
# values_llama = [4.571, 3.77, 2, 4.2]
# values_gemini = [4.4, 4.48, 2.65, 3.2]

# # Number of variables
# N = len(categories)

# # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
# angles = [n / float(N) * 2 * np.pi for n in range(N)]
# angles += angles[:1]

# # Initialise the radar plot
# ax = plt.subplot(111, polar=True)

# # Draw one axe per variable + add labels
# plt.xticks(angles[:-1], categories)

# # Plot each line and add values
# values_gpt += values_gpt[:1]
# ax.plot(angles, values_gpt, linewidth=1, linestyle='solid', label='GPT')
# ax.fill(angles, values_gpt, 'b', alpha=0.1)
# for i, value in enumerate(values_gpt):
#     ax.text(angles[i], value, str(value), ha='center',
#             va='center', color='blue', size=10)

# values_llama += values_llama[:1]
# ax.plot(angles, values_llama, linewidth=1, linestyle='solid', label='LLaMA')
# ax.fill(angles, values_llama, 'r', alpha=0.1)
# for i, value in enumerate(values_llama):
#     ax.text(angles[i], value, str(value), ha='center',
#             va='center', color='red', size=10)

# values_gemini += values_gemini[:1]
# ax.plot(angles, values_gemini, linewidth=1, linestyle='solid', label='Gemini')
# ax.fill(angles, values_gemini, 'g', alpha=0.1)
# for i, value in enumerate(values_gemini):
#     ax.text(angles[i], value, str(value), ha='center',
#             va='center', color='green', size=10)

# # Add legend
# plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

# # Show the plot
# plt.show()
