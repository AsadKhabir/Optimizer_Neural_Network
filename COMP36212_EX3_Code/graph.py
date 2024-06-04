import matplotlib.pyplot as plt

# Updated data for Gearing Ratio
years = ['2019', '2020', '2021', '2022']
nvidia_gearing = [1.37, 0.99, 0.44, 0.21]  # Corrected figures for Nvidia
amd_gearing = [7.03, 6.98, 14.81, 6.99]    # Corrected figures for AMD
intel_gearing = [5.46, 6.19, 89.19, 57.05] # Corrected figures for Intel

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(years, nvidia_gearing, marker='o', label='Nvidia')
plt.plot(years, amd_gearing, marker='o', label='AMD')
plt.plot(years, intel_gearing, marker='o', label='Intel')

# Adding titles and labels
plt.title('Gearing Ratio Over Time')
plt.xlabel('Year')
plt.ylabel('Gearing Ratio')
plt.xticks(ticks=years)  # Ensure all years are displayed
plt.legend()

# Show the graph
plt.grid(True)
plt.show()
